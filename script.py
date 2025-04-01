import torch
import argparse
import os
from gmflow.gmflow import GMFlow  # Ensure gmflow module is available
import cv2
import numpy as np
from torchvision.utils import flow_to_image
import matplotlib.pyplot as plt

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='path/to/pretrained.pth', type=str, help='Path to pretrained model checkpoint')
    parser.add_argument('--image1', default='frame1.png', type=str, help='Path to first image')
    parser.add_argument('--image2', default='frame2.png', type=str, help='Path to second image')
    parser.add_argument('--output_dir', default='output', type=str, help='Directory to save flow output')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--num_scales', default=1, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--attention_type', default='swin', type=str)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    return parser

def load_image(image_path):
    """Load and preprocess an image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
    return img_tensor.unsqueeze(0)  # Add batch dimension

def overlay_flow_on_image(image, flow, alpha=0.5):
    """
    Overlay the flow visualization on the input image.
    Args:
        image: Tensor of shape [B, 3, H, W], values in [0, 1]
        flow: Tensor of shape [B, 2, H, W], optical flow
        alpha: Blending factor (0 to 1), controls the opacity of the flow overlay
    Returns:
        Blended image tensor of shape [B, 3, H, W]
    """
    # Convert flow to RGB image
    flow_rgb = flow_to_image(flow)  # Shape: [B, 3, H, W], values in [-1, 1]
    flow_rgb = (flow_rgb + 1) / 2  # Map to [0, 1] for blending

    # Ensure image is in [0, 1]
    image = torch.clamp(image, 0, 1)

    # Blend the image and flow
    blended = alpha * flow_rgb + (1 - alpha) * image
    return torch.clamp(blended, 0, 1)

def plot(image):
    """Plot a single image using Matplotlib."""
    plt.figure(figsize=(8, 6))
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())  # Convert CHW to HWC
    plt.axis('off')
    plt.title("Input Image with Flow Overlay")
    plt.tight_layout()
    plt.show()

def main(args):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = GMFlow(
        feature_channels=args.feature_channels,
        num_scales=args.num_scales,
        upsample_factor=args.upsample_factor,
        num_head=args.num_head,
        attention_type=args.attention_type,
        ffn_dim_expansion=args.ffn_dim_expansion,
        num_transformer_layers=args.num_transformer_layers,
    ).to(device)

    # Load pretrained weights safely
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    model.eval()

    # Load image pair
    img1_batch = load_image(args.image1).to(device)  # Batch size of 1 for simplicity
    img2 = load_image(args.image2).to(device)

    # Define inference parameters (from original defaults)
    attn_splits_list = [2]  # Number of splits in attention
    corr_radius_list = [-1]  # -1 indicates global matching
    prop_radius_list = [-1]  # -1 indicates global attention

    # Inference
    with torch.no_grad():
        results_dict = model(
            img1_batch, img2,
            attn_splits_list=attn_splits_list,
            corr_radius_list=corr_radius_list,
            prop_radius_list=prop_radius_list
        )
        predicted_flows = results_dict['flow_preds'][-1]  # Final flow prediction

    # Overlay flow on the input image
    blended_image = overlay_flow_on_image(img1_batch, predicted_flows, alpha=0.5)

    # Plot the result
    plot(blended_image)

    # Save the blended image
    os.makedirs(args.output_dir, exist_ok=True)
    blended_img = blended_image.squeeze().permute(1, 2, 0).cpu().numpy() * 255  # Convert to [0, 255]
    cv2.imwrite(os.path.join(args.output_dir, 'flow_overlay.png'), cv2.cvtColor(blended_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"Flow overlay saved to {args.output_dir}/flow_overlay.png")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)