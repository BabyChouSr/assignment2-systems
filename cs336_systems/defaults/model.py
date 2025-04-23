from cs336_systems.benchmark.benchmarking_script import BenchmarkConfig

gpt2_small = BenchmarkConfig(
    d_model=768,
    d_ff=3072,
    num_layers=12,
    num_heads=12,
    vocab_size=10000,
    context_length=512,
    rope_theta=10000,
    warmup_steps=5,
    profile_steps=10,
    profile_pass="forward",
)

gpt2_medium = BenchmarkConfig(
    d_model=1024,
    d_ff=4096,
    num_layers=24,
    num_heads=16,
    vocab_size=10000,
    context_length=512,
    rope_theta=10000,
    warmup_steps=5,
    profile_steps=10,
    profile_pass="forward",
)

gpt2_large = BenchmarkConfig(
    d_model=1280,
    d_ff=5120,
    num_layers=36,
    num_heads=20,
    vocab_size=10000,
    context_length=512,
    rope_theta=10000,
    warmup_steps=5,
    profile_steps=10,
    profile_pass="forward",
)

gpt2_xl = BenchmarkConfig(
    d_model=1600,
    d_ff=6400,
    num_layers=48,
    num_heads=25,
    vocab_size=10000,
    context_length=512,
    rope_theta=10000,
    warmup_steps=5,
    profile_steps=10,
    profile_pass="forward",
)

gpt2_2_7B = BenchmarkConfig(
    d_model=2560,
    d_ff=10240,
    num_layers=32,
    num_heads=32,
    vocab_size=10000,
    context_length=512,
    rope_theta=10000,
    warmup_steps=5,
    profile_steps=10,
    profile_pass="forward",
)