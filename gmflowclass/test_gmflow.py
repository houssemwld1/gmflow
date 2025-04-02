import torch
import unittest
from gmflow.gmflow.gmflowclass.GMFL import GMFlowEstimator  # Absolute import
from PIL import Image
import numpy as np
from gmflow.gmflow.utils.flow_viz import save_vis_flow_tofile

class TestGMFlowEstimator(unittest.TestCase):
    def setUp(self):
        # Set up a GMFlowEstimator instance with default parameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GMFlowEstimator(device=self.device)

    def test_normalize_for_gmflow(self):
        # Test the normalization function
        img = torch.rand(1, 3, 64, 64).to(self.device)  # Random image tensor
        normalized_img = self.model.normalize_for_gmflow(img)
        mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        expected = (img - mean) / std
        self.assertTrue(torch.allclose(normalized_img, expected, atol=1e-6))

    def test_forward_single_direction(self):
        # Test forward pass with single-direction flow
        img1 = torch.rand(1, 3, 64, 64).to(self.device)  # Random first frame
        img2 = torch.rand(1, 3, 64, 64).to(self.device)  # Random second frame
        self.model.pred_bidir_flow = False
        flow = self.model(img1, img2)
        self.assertEqual(flow.shape, (1, 2, 64, 64))  # Check output shape

    def test_forward_with_inference_size(self):
        # Test forward pass with specified inference size
        img1 = torch.rand(1, 3, 64, 64).to(self.device)  # Random first frame
        img2 = torch.rand(1, 3, 64, 64).to(self.device)  # Random second frame
        self.model.inference_size = (32, 32)
        flow = self.model(img1, img2)
        self.assertEqual(flow.shape, (1, 2, 64, 64))  # Output should be resized back to original size

    def test_real_images_and_save_flow(self):
        # Test the model on two real images and save the flow visualization
        img1_path = '/home/houssem/Downloads/gmflow-20250401T174539Z-001/gmflow/gmflow/demo/sintel_market_1/img1_batch.png'
        img2_path = '/home/houssem/Downloads/gmflow-20250401T174539Z-001/gmflow/gmflow/demo/sintel_market_1/img2_batch.png'
        output_flow_path = './flow_output.png'  # Specify a file name for the flow visualization

        # Load and preprocess the images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        img1 = torch.tensor(np.array(img1).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        img2 = torch.tensor(np.array(img2).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        img1, img2 = img1.to(self.device), img2.to(self.device)

        # Load weights and adjust state dictionary keys
        checkpoint = torch.load('../pretrained/gmflow_things-e9887eda.pth', map_location=self.device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        # Add 'model.' prefix to match the model's expected keys
        adjusted_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
        self.model.load_state_dict(adjusted_state_dict)

        # Run the model
        self.model.eval()  # Set to evaluation mode
        self.model.pred_bidir_flow = False
        flow = self.model.forward(img1, img2)

        # Check the output shape
        self.assertEqual(flow.shape[0], 1)  # Ensure batch dimension is present
        self.assertEqual(flow.shape[1], 2)  # Ensure 2 channels for flow (x, y)

        # Convert flow to [H, W, 2] format for visualization
        flow_np = flow[0].permute(1, 2, 0).cpu().numpy()  # Shape: [H, W, 2]

        # Save the flow visualization to a file
        save_vis_flow_tofile(flow_np, output_flow_path)

if __name__ == '__main__':
    unittest.main()