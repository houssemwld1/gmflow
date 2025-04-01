import torch
import argparse
import os
import cv2
import numpy as np
from gmflow.gmflow import GMFlow
from torchvision.utils import flow_to_image
import matplotlib.pyplot as plt

def get_args_parser():
    """Define command-line arguments for inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to pretrained model checkpoint')
    parser.add_argument('--image1', type=str, help='Path to the first image')
    parser.add_argument('--image2', type=str, help='Path to the second image')
    parser.add_argument('--inference_dir', type=str, help='Directory containing image pairs for inference')
    parser.add_argument('--output_dir', default='output', type=str, help='Directory to save flow output')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--num_scales', default=1, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--attention_type', default='swin', type=str)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+')
    parser.add_argument('--save_vis_flow', action='store_true', help='Save flow visualization')
    return parser

def load_image(image_path):
    """Load and preprocess an image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
    return img_tensor.unsqueeze(0)  # Add batch dimension

def overlay_flow_on_image(image, flow, alpha=0.5):
    """Overlay the flow visualization on the input image."""
    flow_rgb = flow_to_image(flow)  # Shape: [B, 3, H, W], values in [-1, 1]
    flow_rgb = (flow_rgb + 1) / 2  # Map to [0, 1]
    image = torch.clamp(image, 0, 1)
    blended = alpha * flow_rgb + (1 - alpha) * image
    return torch.clamp(blended, 0, 1)

def plot(image):
    """Plot a single image using Matplotlib."""
    image = image.squeeze(0)  # Remove batch dimension
    plt.figure(figsize=(8, 6))
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())  # Convert CHW to HWC
    plt.axis('off')
    plt.title("Input Image with Flow Overlay")
    plt.tight_layout()
    plt.show()

def main(args):
    """Perform inference using the GMFlow model."""
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

    # Load pretrained weights
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    model.eval()

    if args.inference_dir:
        # Inference on a directory of image pairs
        image_pairs = []  # List of (img1_path, img2_path) pairs
        # Note: Populate `image_pairs` based on your directory structure (e.g., sorted file list)
        for img1_path, img2_path in image_pairs:
            img1 = load_image(img1_path).to(device)
            img2 = load_image(img2_path).to(device)
            with torch.no_grad():
                results_dict = model(
                    img1, img2,
                    attn_splits_list=args.attn_splits_list,
                    corr_radius_list=args.corr_radius_list,
                    prop_radius_list=args.prop_radius_list
                )
                flow_preds = results_dict['flow_preds'][-1]  # Final flow prediction

            if args.save_vis_flow:
                blended_image = overlay_flow_on_image(img1, flow_preds)
                os.makedirs(args.output_dir, exist_ok=True)
                blended_img = blended_image.squeeze().permute(1, 2, 0).cpu().numpy() * 255
                output_path = os.path.join(args.output_dir, f'flow_{os.path.basename(img1_path)}.png')
                cv2.imwrite(output_path, cv2.cvtColor(blended_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
                print(f"Saved flow visualization to {output_path}")
    else:
        # Inference on a single image pair
        img1 = load_image(args.image1).to(device)
        img2 = load_image(args.image2).to(device)
        with torch.no_grad():
            results_dict = model(
                img1, img2,
                attn_splits_list=args.attn_splits_list,
                corr_radius_list=args.corr_radius_list,
                prop_radius_list=args.prop_radius_list
            )
            flow_preds = results_dict['flow_preds'][-1]  # Final flow prediction

        # Overlay flow on the input image and visualize
        blended_image = overlay_flow_on_image(img1, flow_preds)
        plot(blended_image)

        if args.save_vis_flow:
            os.makedirs(args.output_dir, exist_ok=True)
            blended_img = blended_image.squeeze().permute(1, 2, 0).cpu().numpy() * 255
            output_path = os.path.join(args.output_dir, 'flow_overlay.png')
            cv2.imwrite(output_path, cv2.cvtColor(blended_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
            print(f"Saved flow visualization to {output_path}")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)