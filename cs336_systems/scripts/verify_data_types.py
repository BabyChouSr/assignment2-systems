import torch
from cs336_systems.modeling.toy_model import ToyModel

model = ToyModel(5, 5).to("cuda")
batch = torch.randn(3, 5, dtype=torch.float32, device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    for p in model.parameters():
        print(f"Model dtype in autocast context: {p.data.dtype}")
    logits = model(batch)
    print(f"Output logits dtype: {logits.dtype}")
    loss = logits.mean()
    print(f"Loss dtype: {loss.dtype}")
    loss.backward()

    for p in model.parameters():
        print(f"gradient type: {p.grad.data.dtype}")
