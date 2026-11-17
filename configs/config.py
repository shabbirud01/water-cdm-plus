import os


class Config:
    # Diffusion
    T = 200                         # was 1000
    BETA_START = 1e-4
    BETA_END = 0.02

    # Model — CPU-friendly lightweight architecture
    BASE_CHANNELS = 32              # was 64
    CHANNEL_MULTS = [1, 2, 4]      # was [1, 2, 4, 8]
    NUM_ATTENTION_HEADS = 2         # was 8
    NUM_RESTORMER_BLOCKS = 1        # was 2

    # Training — CPU-scale
    IMAGE_SIZE = 128                # was 256
    BATCH_SIZE = 1                  # was 4
    LR = 1e-3                      # was 1e-4
    LR_MIN = 1e-5
    NUM_ITERS = 20000              # was 200_000
    SAVE_EVERY = 5000              # was 10_000
    LOG_EVERY = 50                 # was 100

    # Loss weights
    W_MSE = 1.0
    W_PERCEPTUAL = 0.1
    W_SSIM = 0.1

    # DDIM
    DDIM_STEPS = 50                # restored from original
    DDIM_ETA = 0.0

    # AMP — disabled, requires CUDA
    USE_AMP = False

    # Dataset
    DATASET = "UIEB"
    DATA_DIR = "data/UIEB"
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    RESULTS_DIR = "results"

    def ensure_dirs(self):
        for d in [self.CHECKPOINT_DIR, self.LOG_DIR, self.RESULTS_DIR]:
            os.makedirs(d, exist_ok=True)
