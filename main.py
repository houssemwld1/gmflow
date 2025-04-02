import torch
import argparse
import os
import numpy as np
from PIL import Image

from gmflow.gmflow import GMFlow
from evaluate import inference_on_dir
from utils import misc


def get_args_parser():
    parser = argparse.ArgumentParser()

    # inference on a directory
    parser.add_argument('--inference_dir', default=None, type=str, required=True,
                        help='Directory containing images for inference')
    parser.add_argument('--output_path', default='output', type=str,
                        help='Where to save the inference results')
    parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                        help='Specify the inference size')
    parser.add_argument('--dir_paired_data', action='store_true',
                        help='Paired data in a directory instead of a sequence')
    parser.add_argument('--save_flo_flow', action='store_true',
                        help='Save flow as .flo files')
    parser.add_argument('--save_image_flow', action='store_true',
                        help='Save flow as image files')
    parser.add_argument('--pred_bidir_flow', action='store_true',
                        help='Predict bidirectional flow')
    parser.add_argument('--fwd_bwd_consistency_check', action='store_true',
                        help='Forward-backward consistency check with bidirectional flow')

    # GMFlow model
    parser.add_argument('--num_scales', default=1, type=int,
                        help='Number of scales for GMFlow')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--attention_type', default='swin', type=str)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)

    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='Number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='Correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='Self-attention radius for flow propagation, -1 indicates global attention')

    # resume pretrained model
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to the pretrained model checkpoint')

    return parser


def save_flow_as_image(flow, output_path):
    """Save optical flow as an image."""
    flow_magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    flow_magnitude = (flow_magnitude / flow_magnitude.max() * 255).astype(np.uint8)
    flow_image = Image.fromarray(flow_magnitude)
    flow_image.save(output_path)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = GMFlow(feature_channels=args.feature_channels,
                   num_scales=args.num_scales,
                   upsample_factor=args.upsample_factor,
                   num_head=args.num_head,
                   attention_type=args.attention_type,
                   ffn_dim_expansion=args.ffn_dim_expansion,
                   num_transformer_layers=args.num_transformer_layers,
                   ).to(device)

    if args.resume:
        print('Loading checkpoint:', args.resume)
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)

    model.eval()

    # Perform inference
    inference_results = inference_on_dir(model,
                                         inference_dir=args.inference_dir,
                                         output_path=args.output_path,
                                         padding_factor=16,  # Default padding factor
                                         inference_size=args.inference_size,
                                         paired_data=args.dir_paired_data,
                                         save_flo_flow=args.save_flo_flow,
                                         attn_splits_list=args.attn_splits_list,
                                         corr_radius_list=args.corr_radius_list,
                                         prop_radius_list=args.prop_radius_list,
                                         pred_bidir_flow=args.pred_bidir_flow,
                                         fwd_bwd_consistency_check=args.fwd_bwd_consistency_check)

    # Save flow as images if required




if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
