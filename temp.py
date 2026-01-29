from spectre import SpectreImageFeatureExtractor, MODEL_CONFIGS
import torch

config = MODEL_CONFIGS['spectre-large-pretrained']
model = SpectreImageFeatureExtractor.from_config(config)
model.eval()

# Dummy input: (batch, crops, channels, height, width, depth)
# For a (3 x 3 x 4) grid of (128 x 128 x 64) CT patches -> Total scan size (384 x 384 x 256)
x = torch.randn(1, 96, 1, 128, 128, 64)
with torch.no_grad():
    features = model(x, grid_size=(4,4,6))
print("Features shape:", features.shape)