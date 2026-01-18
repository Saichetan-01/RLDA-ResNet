import torch
import numpy as np
from sklearn.cluster import KMeans
import torchvision
import torch.nn.functional as F

def extract_centroids_and_dot_products(model, layer_name, num_centroids=32, device='cpu'):
    print(f" Extracting LUTs for: {layer_name}...")

    # Get convolution layer weights
    conv_layer = dict(model.named_modules())[layer_name]
    W = conv_layer.weight.data.to(device)  # (out_ch, in_ch, kH, kW)
    out_ch, in_ch, kH, kW = W.shape

    # Hook to capture input feature maps
    activation = {}
    def hook(mod, inp, out):
        activation['features'] = inp[0].detach()
    hook_handle = conv_layer.register_forward_hook(hook)

    #  Dynamically adjust collection sizes & centroids per layer
    if "layer4" in layer_name:
        num_batches = 50  # More samples for stability
        num_centroids = 16
    elif "layer3" in layer_name:
        num_batches = 30
        num_centroids = 24
    else:
        num_batches = 20

    batch_size = 32
    patches_list = []
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)

    for _ in range(num_batches):
        model(dummy_input)
        feats = activation['features']
        patches = F.unfold(feats, kernel_size=(kH, kW), padding=conv_layer.padding, stride=conv_layer.stride)
        patches = patches.permute(0, 2, 1).reshape(-1, in_ch * kH * kW).cpu().numpy()
        patches_list.append(patches)

    hook_handle.remove()

    #  Aggregate all collected patches
    patches = np.concatenate(patches_list, axis=0)
    print(f" Total extracted patches shape: {patches.shape} for {layer_name}")

    #  K-Means Clustering for Centroids
    kmeans_xhat = KMeans(n_clusters=num_centroids, random_state=0).fit(patches)
    centroids_xhat = kmeans_xhat.cluster_centers_

    #  Compute residuals (x - x̂)
    labels_xhat = kmeans_xhat.predict(patches)
    residuals = patches - centroids_xhat[labels_xhat]

    #  Cluster residuals
    kmeans_residual = KMeans(n_clusters=num_centroids, random_state=0).fit(residuals)
    residual_centroids = kmeans_residual.cluster_centers_

    #  Compute Dot Products
    W_flat = W.view(out_ch, -1).cpu().numpy()
    dot_products_xhat = np.dot(centroids_xhat, W_flat.T)
    dot_products_residual = np.dot(residual_centroids, W_flat.T)

    #  Store LUT
    lut = {
        'centroids': torch.tensor(centroids_xhat, dtype=torch.float32),
        'residual_centroids': torch.tensor(residual_centroids, dtype=torch.float32),
        'dot_centroids': torch.tensor(dot_products_xhat, dtype=torch.float32),
        'dot_residual_centroids': torch.tensor(dot_products_residual, dtype=torch.float32)
    }

    #  Save LUT file
    lut_filename = f'lut_{layer_name.replace(".", "_")}.pth'
    torch.save(lut, lut_filename)
    print(f" LUT saved successfully: {lut_filename}\n")

#  Running this extraction for all ResNet-18 layers
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    state_dict = torch.load('trained_resnet18_cifar10.pth', map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    #  List of convolutional layers
    conv_layers = [
        'layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', 'layer1.1.conv2',
        'layer2.0.conv1', 'layer2.0.conv2', 'layer2.1.conv1', 'layer2.1.conv2',
        'layer3.0.conv1', 'layer3.0.conv2', 'layer3.1.conv1', 'layer3.1.conv2',
        'layer4.0.conv1', 'layer4.0.conv2', 'layer4.1.conv1', 'layer4.1.conv2'
    ]

    for layer_name in conv_layers:
        extract_centroids_and_dot_products(model, layer_name, num_centroids=32, device=device)
