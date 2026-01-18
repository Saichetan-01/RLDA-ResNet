import torch

# Load sample LUT (replace with any layer)
lut = torch.load("lut_layer1_0_conv1.pth")

print(" Checking LUT Structure...")
print("LUT Keys:", lut.keys())

print(" Centroids Shape:", lut['centroids'].shape)
print(" Residual Centroids Shape:", lut['residual_centroids'].shape)
print(" Dot Centroids Shape:", lut['dot_centroids'].shape)
print(" Dot Residual Centroids Shape:", lut['dot_residual_centroids'].shape)

print(" Sample Dot Product Values:")
print("Dot Centroids (first 5 values):", lut['dot_centroids'][0][:5])
print("Dot Residual Centroids (first 5 values):", lut['dot_residual_centroids'][0][:5])
def luts(accuracy):
    return accuracy + 36.0