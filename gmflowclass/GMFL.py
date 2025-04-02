import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gmflow.gmflow.gmflow.gmflow import GMFlow  # Relative import within the gmflow module
from gmflow.gmflow.gmflow.geometry import forward_backward_consistency_check 
from gmflow.gmflow.utils.utils import InputPadder
import os
from glob import glob
from PIL import Image
from gmflow.gmflow.utils import frame_utils

class GMFlowEstimator(nn.Module):
    def __init__(self, 
                 feature_channels=128,
                 num_scales=1,
                 upsample_factor=8,
                 num_head=1,
                 attention_type='swin',
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 attn_splits_list=[2],
                 corr_radius_list=[-1],
                 prop_radius_list=[-1],
                 pred_bidir_flow=False,
                 fwd_bwd_consistency_check=False,
                 padding_factor=8,
                 inference_size=None,
                 resume=None,
                 device='cuda'):
        super(GMFlowEstimator, self).__init__()
        
        # GMFlow model configuration
        self.model = GMFlow(
            feature_channels=feature_channels,
            num_scales=num_scales,
            upsample_factor=upsample_factor,
            num_head=num_head,
            attention_type=attention_type,
            ffn_dim_expansion=ffn_dim_expansion,
            num_transformer_layers=num_transformer_layers
        ).to(device)

        # Load pretrained weights if provided
        if resume:
            print(f'Loading GMFlow checkpoint: {resume}')
            checkpoint = torch.load(resume, map_location=device)
            self.model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)

        # Inference settings
        self.attn_splits_list = attn_splits_list
        self.corr_radius_list = corr_radius_list
        self.prop_radius_list = prop_radius_list
        self.pred_bidir_flow = pred_bidir_flow
        self.fwd_bwd_consistency_check = fwd_bwd_consistency_check
        self.padding_factor = padding_factor
        self.inference_size = inference_size
        self.device = device

        # Set model to evaluation mode
        self.model.eval()

    def normalize_for_gmflow(self, img):
        """Normalize image for GMFlow: subtract mean [0.5, 0.5, 0.5] and divide by std [0.5, 0.5, 0.5]."""
        mean = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(1, 3, 1, 1)
        return (img - mean) / std

    def forward(self, img1, img2):
        """
        Compute optical flow between img1 and img2.
        Args:
            img1 (torch.Tensor): First frame, shape [B, 3, H, W], values in [0, 1]
            img2 (torch.Tensor): Second frame, shape [B, 3, H, W], values in [0, 1]
        Returns:
            If pred_bidir_flow=False: torch.Tensor, optical flow [B, 2, H, W]
            If pred_bidir_flow=True: tuple of (flow_forward, flow_backward), each [B, 2, H, W]
        """
        if self.fwd_bwd_consistency_check:
            assert self.pred_bidir_flow, "Forward-backward consistency check requires bidirectional flow"

        # Ensure inputs are on the correct device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        # Normalize inputs
        img1_norm = self.normalize_for_gmflow(img1)
        img2_norm = self.normalize_for_gmflow(img2)

        # Padding
        if self.inference_size is None:
            padder = InputPadder(img1.shape, padding_factor=self.padding_factor)
            img1_norm, img2_norm = padder.pad(img1_norm, img2_norm)
        else:
            img1_norm, img2_norm = img1_norm, img2_norm

        # Resize if inference_size is specified
        if self.inference_size is not None:
            assert isinstance(self.inference_size, (list, tuple))
            ori_size = img1_norm.shape[-2:]
            img1_norm = F.interpolate(img1_norm, size=self.inference_size, mode='bilinear', align_corners=True)
            img2_norm = F.interpolate(img2_norm, size=self.inference_size, mode='bilinear', align_corners=True)

        # Compute optical flow
        with torch.no_grad():
            results_dict = self.model(
                img1_norm,
                img2_norm,
                attn_splits_list=self.attn_splits_list,
                corr_radius_list=self.corr_radius_list,
                prop_radius_list=self.prop_radius_list,
                pred_bidir_flow=self.pred_bidir_flow
            )
            flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W] or [2, B, 2, H, W] if pred_bidir_flow=True

        # Resize back if inference_size was used
        if self.inference_size is not None:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear', align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / self.inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / self.inference_size[-2]

        # Unpad if padding was applied
        if self.inference_size is None:
            flow_pr = padder.unpad(flow_pr)

        # Handle bidirectional flow
        if self.pred_bidir_flow:
            flow_forward = flow_pr[0]  # [B, 2, H, W]
            flow_backward = flow_pr[1]  # [B, 2, H, W]

            # Forward-backward consistency check
            if self.fwd_bwd_consistency_check:
                fwd_occ, bwd_occ = forward_backward_consistency_check(flow_forward.unsqueeze(0), flow_backward.unsqueeze(0))
                # Optionally, you can use fwd_occ and bwd_occ to mask unreliable flow regions
                # For simplicity, we'll return the flows as-is

            return flow_forward, flow_backward
        else:
            return flow_pr  # [B, 2, H, W]

    def inference_on_dir(self, inference_dir, output_path='output', paired_data=False, save_flo_flow=False):
        """
        Perform inference on all consecutive image pairs in a directory.
        
        Args:
            inference_dir (str): Path to directory containing images
            output_path (str): Path to save output flow visualizations
            paired_data (bool): If True, treats directory as containing paired test data
            save_flo_flow (bool): If True, saves flow in .flo format for quantitative evaluation
            
        Returns:
            list: List of tuples containing (flow, output_file_path) for each processed pair
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        filenames = sorted(glob(os.path.join(inference_dir, '*')))
        print('%d images found' % len(filenames))

        stride = 2 if paired_data else 1

        if paired_data:
            assert len(filenames) % 2 == 0

        results = []
        
        for test_id in range(0, len(filenames) - 1, stride):
            # Read and process images
            image1 = frame_utils.read_gen(filenames[test_id])
            image2 = frame_utils.read_gen(filenames[test_id + 1])

            image1 = np.array(image1).astype(np.uint8)
            image2 = np.array(image2).astype(np.uint8)

            # Handle grayscale images
            if len(image1.shape) == 2:  # gray image
                image1 = np.tile(image1[..., None], (1, 1, 3))
                image2 = np.tile(image2[..., None], (1, 1, 3))
            else:
                image1 = image1[..., :3]
                image2 = image2[..., :3]

            # Convert to tensor and normalize
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float() / 255.0
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float() / 255.0

            # Compute flow
            if self.pred_bidir_flow:
                flow_forward, flow_backward = self.forward(image1.unsqueeze(0), image2.unsqueeze(0))
                flow_pr = torch.stack([flow_forward, flow_backward])
            else:
                flow_pr = self.forward(image1.unsqueeze(0), image2.unsqueeze(0))

            # Convert flow to numpy
            flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

            # Save visualization
            base_name = os.path.basename(filenames[test_id])[:-4]
            output_file = os.path.join(output_path, f'{base_name}_flow.png')
            self.save_vis_flow_tofile(flow, output_file)

            # Handle bidirectional flow
            if self.pred_bidir_flow:
                # Save backward flow visualization
                flow_bwd = flow_pr[1].permute(1, 2, 0).cpu().numpy()
                output_file_bwd = os.path.join(output_path, f'{base_name}_flow_bwd.png')
                self.save_vis_flow_tofile(flow_bwd, output_file_bwd)

                # Forward-backward consistency check
                if self.fwd_bwd_consistency_check:
                    fwd_occ, bwd_occ = forward_backward_consistency_check(
                        flow_pr[0].unsqueeze(0), flow_pr[1].unsqueeze(0))
                    
                    # Save occlusion maps
                    fwd_occ_file = os.path.join(output_path, f'{base_name}_occ.png')
                    bwd_occ_file = os.path.join(output_path, f'{base_name}_occ_bwd.png')
                    
                    Image.fromarray((fwd_occ[0].cpu().numpy() * 255.).astype(np.uint8)).save(fwd_occ_file)
                    Image.fromarray((bwd_occ[0].cpu().numpy() * 255.).astype(np.uint8)).save(bwd_occ_file)

            # Save .flo file if requested
            if save_flo_flow:
                flo_file = os.path.join(output_path, f'{base_name}_pred.flo')
                frame_utils.writeFlow(flo_file, flow)
                results.append((flow, flo_file))
            else:
                results.append((flow, output_file))

        return results

    @staticmethod
    def save_vis_flow_tofile(flow, output_file):
        """
        Save flow visualization to file.
        
        Args:
            flow (np.ndarray): Optical flow array [H, W, 2]
            output_file (str): Path to save visualization
        """
        flow_img = frame_utils.flow_to_image(flow)
        Image.fromarray(flow_img).save(output_file)