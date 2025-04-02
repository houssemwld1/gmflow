import torch
import unittest
from gmflow.gmflow.gmflowclass.GMFL import GMFlowEstimator  # Absolute import
from PIL import Image
import numpy as np
import os 
from gmflow.gmflow.utils.flow_viz import save_vis_flow_tofile
class TestGMFlowEstimator(unittest.TestCase):
    def setUp(self):
        # Set up a GMFlowEstimator instance with default parameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resume_path = '../pretrained/gmflow_things-e9887eda.pth'  # Adjust path as needed
        if not os.path.exists(self.resume_path):
            self.skipTest(f"Pretrained checkpoint not found at {self.resume_path}")

        self.model = GMFlowEstimator(
            device=self.device,
            resume=self.resume_path,
            pred_bidir_flow=False,  # Test unidirectional flow
            fwd_bwd_consistency_check=False,
            padding_factor=16,
            inference_size=None
        )

    def test_real_images_and_save_flow(self):
        # Test the model on two real images and save the flow visualization
        img1_path = '../demo/sintel_market_1/img1_batch.png'
        img2_path = '../demo/sintel_market_1/img2_batch.png'
        output_flow_path = './flow_output.png'

        # Check if image files exist
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            self.skipTest(f"Image files not found: {img1_path}, {img2_path}")

        # Load and preprocess the images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        img1 = torch.tensor(np.array(img1).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0  # [1, 3, H, W]
        img2 = torch.tensor(np.array(img2).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0  # [1, 3, H, W]
        img1, img2 = img1.to(self.device), img2.to(self.device)

        # Run the model
        self.model.eval()  # Ensure evaluation mode
        flow = self.model.forward(img1, img2)

        # Check the output shape
        self.assertEqual(flow.shape[0], 1, "Batch dimension should be 1")
        self.assertEqual(flow.shape[1], 2, "Flow should have 2 channels (x, y)")
        self.assertEqual(flow.shape[2], img1.shape[2], "Flow height should match input height")
        self.assertEqual(flow.shape[3], img1.shape[3], "Flow width should match input width")

        # Convert flow to [H, W, 2] format for visualization
        flow_np = flow[0].permute(1, 2, 0).cpu().numpy()  # Shape: [H, W, 2]

        # Save the flow visualization to a file
        save_vis_flow_tofile(flow_np, output_flow_path)
        self.assertTrue(os.path.exists(output_flow_path), f"Flow visualization file {output_flow_path} was not created")

    def tearDown(self):
        # Clean up VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    unittest.main()
# class TestGMFlowEstimator(unittest.TestCase):
#     def setUp(self):
#         # Set up a GMFlowEstimator instance with default parameters
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.model = GMFlowEstimator(device=self.device, resume='../pretrained/gmflow_things-e9887eda.pth')

#     def test_real_images_and_save_flow(self):
#         # Test the model on two real images and save the flow visualization
#         img1_path = '../demo/sintel_market_1/img1_batch.png'
#         img2_path = '../demo/sintel_market_1/img2_batch.png'
#         output_flow_path = './flow_output.png'  # Specify a file name for the flow visualization

#         # Load and preprocess the images
#         img1 = Image.open(img1_path).convert('RGB')
#         img2 = Image.open(img2_path).convert('RGB')
#         img1 = torch.tensor(np.array(img1).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
#         img2 = torch.tensor(np.array(img2).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
#         img1, img2 = img1.to(self.device), img2.to(self.device)

#         # Run the model
#         self.model.eval()  # Set to evaluation mode
#         self.model.pred_bidir_flow = False
#         flow = self.model.forward(img1, img2)
#         # Check the output shape
#         self.assertEqual(flow.shape[0], 1)  # Ensure batch dimension is present
#         self.assertEqual(flow.shape[1], 2)  # Ensure 2 channels for flow (x, y)

#         # Convert flow to [H, W, 2] format for visualization
#         flow_np = flow[0].permute(1, 2, 0).cpu().numpy()  # Shape: [H, W, 2]

#         # Save the flow visualization to a file
#         save_vis_flow_tofile(flow_np, output_flow_path)

# if __name__ == '__main__':
#     unittest.main()