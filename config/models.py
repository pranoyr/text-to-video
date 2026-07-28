# Common configurations for all models
BASE_KWARGS = {
    "patch_size": 2,
    "dim_cond": 4096,  
}

# Approx 80M parameters
SMALL_CONFIG = {
    **BASE_KWARGS,
    "dim": 640,
    "depth": 16,
    "heads": 10,
    "dim_head": 64,
    "mlp_dim": 2560,
}

# Approx 1.2B parameters
MEDIUM_CONFIG = {
    **BASE_KWARGS,
    "dim": 1792,
    "depth": 32,
    "heads": 28,
    "dim_head": 64,
    "mlp_dim": 7168,
}


MODEL_CONFIGS = {
    "small": SMALL_CONFIG,
    "medium": MEDIUM_CONFIG,
}
