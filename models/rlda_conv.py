import torch
import torch.nn as nn
import torch.nn.functional as F

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class RLDAConv(nn.Module):
    def __init__(self, lut_path, in_channels, out_channels, kernel_size, stride=1, padding=1, use_bn=True):
        super().__init__()
        # Load LUT from file.
        # Expected keys: 'centroids', 'residual_centroids', 'dot_centroids', 'dot_residual_centroids'
        lut = torch.load(lut_path)
        self.register_buffer('centroids', lut['centroids'])  # e.g. shape: [32, D]
        self.register_buffer('residual_centroids', lut['residual_centroids'])  # e.g. shape: [32, D]
        self.register_buffer('dot_centroids', lut['dot_centroids'])  # e.g. shape: [32, out_channels]
        self.register_buffer('dot_residual_centroids', lut['dot_residual_centroids'])  # e.g. shape: [32, out_channels]

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Ensure tuples for kernel, stride, padding
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        self.use_bn = use_bn
        if self.use_bn:
            # BatchNorm2d for the output channels.
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        bsz, _, h_in, w_in = x.shape
        kH, kW = self.kernel_size

        # Extract patches using F.unfold.
        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        patches = patches.permute(0, 2, 1)  # shape: [bsz, num_patches, D]

        # Use loaded centroids and residual centroids (assumed to be on the correct device)
        centroids = self.centroids
        residual_centroids = self.residual_centroids

        # Nearest neighbor search using Manhattan (L1) distance.
        dist_xhat = torch.cdist(patches, centroids, p=1)
        indices_xhat = dist_xhat.argmin(dim=2)  # [bsz, num_patches]

        # Compute residuals.
        residuals = patches - centroids[indices_xhat]
        # Nearest neighbor search on residual centroids.
        dist_residual = torch.cdist(residuals, residual_centroids, p=1)
        indices_residual = dist_residual.argmin(dim=2)  # [bsz, num_patches]

        # Retrieve pre-computed dot products using the indices.
        dot_xhat = self.dot_centroids[indices_xhat]  # shape: [bsz, num_patches, out_channels]
        dot_residual = self.dot_residual_centroids[indices_residual]  # shape: [bsz, num_patches, out_channels]

        # Sum the dot products to approximate the convolution result.
        output_vals = dot_xhat + dot_residual  # [bsz, num_patches, out_channels]

        # Compute output spatial dimensions.
        outH = (h_in + 2 * self.padding[0] - kH) // self.stride[0] + 1
        outW = (w_in + 2 * self.padding[1] - kW) // self.stride[1] + 1

        # Reshape the output tensor to [bsz, out_channels, outH, outW].
        output = output_vals.permute(0, 2, 1).reshape(bsz, self.out_channels, outH, outW)

        # Apply Batch Normalization if enabled.
        if self.use_bn:
            output = self.bn(output)

        # Wrap with STE to allow gradient flow through non-differentiable parts.
        output = STE.apply(output)
        return output
