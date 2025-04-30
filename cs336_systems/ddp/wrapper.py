import torch
import torch.nn as nn
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class DDP(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        self.world_size = dist.get_world_size()

        # Broadcast initial parameters from rank 0 to ensure all models start identically
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Register hooks for gradient synchronization
        for param in self.module.parameters():
            if param.requires_grad:
                # Use a closure to capture 'param' for each hook
                param.register_post_accumulate_grad_hook(self._make_grad_hook())

    def _make_grad_hook(self):
        # The hook is called after the gradient for this param is accumulated
        def hook(param):
            # grad is the gradient tensor for this param
            # gloo doesn't support op ReduceOp.AVG so we use SUM then we divide by world size after
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append((param, handle))

        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        # Wait for all async all_reduce ops to finish
        for param, handle in self.handles:
            handle.wait()
            # We must divide by world_size to get the average since 
            # gloo doesn't support op ReduceOp.AVG
            if param.grad is not None:
                param.grad.div_(self.world_size)
        self.handles.clear()
    
class DDPBucketed(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()

        self.buckets = []
        self.handles = []
        self.param_to_bucket_idx = {}
        self.world_size = dist.get_world_size()
        self.module = module
        self.bucket_ready_count = []

        # Broadcast initial parameters from rank 0 to ensure all models start identically
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Register hooks for gradient synchronization
        curr_bucket = []  # Bucket holds each param that are in it
        total_mb_seen = 0
        bucket_bytes = bucket_size_mb * 1024 * 1024

        for param in reversed(list(self.module.parameters())):
            if param.requires_grad:
                param_bytes = param.nelement() * param.element_size()
                # dist.breakpoint(rank=0)
                # Handle the case where param_bytes > bucket_bytes and curr_bucket is empty
                # BIG ISSUE - create own bucket but curr_bucket still exists so len(self.buckets)
                # can conflict
                # if param_bytes > bucket_bytes:
                #     # Put this param in its own bucket
                #     solo_bucket = [param]
                #     param.register_post_accumulate_grad_hook(self._make_grad_hook())
                #     self.param_to_bucket_idx[param] = len(self.buckets)
                #     self.buckets.append(solo_bucket)
                param.register_post_accumulate_grad_hook(self._make_grad_hook())
                self.param_to_bucket_idx[param] = len(self.buckets)
                curr_bucket.append(param)
                total_mb_seen += param_bytes

                if total_mb_seen > bucket_bytes:
                    self.buckets.append(curr_bucket)
                    curr_bucket = []
                    total_mb_seen = 0

                # if total_mb_seen + param_bytes >= bucket_bytes and len(curr_bucket) > 0:
                #     last_param = curr_bucket[-1]
                #     last_param.register_post_accumulate_grad_hook(self._make_grad_hook())
                #     self.param_to_bucket_idx[last_param] = len(self.buckets)
                #     self.buckets.append(curr_bucket)
                #     curr_bucket = []
                #     total_mb_seen = 0
                #     curr_bucket.append(param)
                #     total_mb_seen += param_bytes
                # else:
                #     param.register_post_accumulate_grad_hook(self._make_grad_hook())
                #     self.param_to_bucket_idx[param] = len(self.buckets)
                #     curr_bucket.append(param)
                #     total_mb_seen += param_bytes

        if len(curr_bucket) > 0:
            # last_param = curr_bucket[-1]
            # last_param.register_post_accumulate_grad_hook(self._make_grad_hook())
            # self.param_to_bucket_idx[last_param] = len(self.buckets)
            self.buckets.append(curr_bucket)

        self.bucket_ready_count = [0 for i in range(len(self.buckets))]
        
    def _make_grad_hook(self):
        def hook(param):
            # grad is the gradient tensor for this param
            # gloo doesn't support op ReduceOp.AVG so we use SUM then we divide by world size after
            bucket_idx = self.param_to_bucket_idx[param]

            self.bucket_ready_count[bucket_idx] += 1

            if self.bucket_ready_count[bucket_idx] == len(self.buckets[bucket_idx]):
                grads = [param.grad for param in self.buckets[bucket_idx] if param.grad is not None]
                if grads:
                    flattened_grads = _flatten_dense_tensors(grads)
                    handle = dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM, async_op=True)
                    self.handles.append((bucket_idx, handle, flattened_grads))

                    self.bucket_ready_count[bucket_idx] = 0
    
        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        # Wait for all async all_reduce ops to finish
        for bucket_idx, handle, flattened_grads in self.handles:
            handle.wait()
            # We must divide by world_size to get the average since 
            # gloo doesn't support op ReduceOp.AVG
            unflattened_grads = _unflatten_dense_tensors(flattened_grads, self.buckets[bucket_idx])
            for i, param in enumerate(self.buckets[bucket_idx]):
                if param.grad is not None:
                    param.grad.copy_(unflattened_grads[i]).div_(self.world_size)

        self.handles.clear()

class DDPOptim(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.world_size = dist.get_world_size()
        self.local_rank = dist.get_rank()

        # Assume params is a list of parameters in the same order on all ranks
        self.params = list(params)
        self.local_params = [
            param for i, param in enumerate(self.params) if i % self.world_size == self.local_rank
        ]
        self.optimizer = optimizer_cls(self.local_params, **kwargs)
        super().__init__(self.params, self.optimizer.defaults)

    def step(self, closure=None, **kwargs):
        # Step only local parameters
        loss = self.optimizer.step(closure, **kwargs)

        # Synchronize all parameters across ranks
        for i, param in enumerate(self.params):
            owner = i % self.world_size
            # All ranks must wait for the broadcast to finish before proceeding
            dist.broadcast(param.data, src=owner, async_op=False)
            # After broadcast, ensure all param.data are identical across ranks

        return loss

    def add_param_group(self, param_group):
        params = param_group["params"]
        local_params = [p for i, p in enumerate(params) if i % self.world_size == self.local_rank]

        # Add the local params to the optimizer
        super().add_param_group({"params": local_params})

        # Also add the new params to self.params so that future broadcasts include them
        # (Assume param_group["params"] is in the same order on all ranks)
        self.params.extend([p for p in params if not any(p is lp for lp in self.params)])
