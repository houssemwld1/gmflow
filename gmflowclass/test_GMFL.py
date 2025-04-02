import unittest
import torch
import os
import shutil
import numpy as np
from PIL import Image
from gmflow.gmflow.gmflowclass.GMFL import GMFlowEstimator

class TestGMFlowEstimator(unittest.TestCase):
    def setUp(self):
        # Set up a GMFlowEstimator instance
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GMFlowEstimator(device=self.device)

        # Create dummy input tensors
        self.img1 = torch.rand(1, 3, 128, 128).to(self.device)  # [B, C, H, W]
        self.img2 = torch.rand(1, 3, 128, 128).to(self.device)

        # Create a temporary directory for testing inference_on_dir
        self.test_dir = 'test_inference_dir'
        self.output_dir = 'test_output_dir'
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Generate dummy images for testing
        for i in range(2):
            img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(self.test_dir, f'image_{i}.png'))

    def tearDown(self):
        # Clean up temporary directories
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_forward(self):
        # Test the forward method
        flow = self.model.forward(self.img1, self.img2)
        self.assertIsInstance(flow, torch.Tensor)
        self.assertEqual(flow.shape, (1, 2, 128, 128))  # [B, 2, H, W]

    def test_forward_bidirectional(self):
        # Test the forward method with bidirectional flow
        self.model.pred_bidir_flow = True
        flow_forward, flow_backward = self.model.forward(self.img1, self.img2)
        self.assertIsInstance(flow_forward, torch.Tensor)
        self.assertIsInstance(flow_backward, torch.Tensor)
        self.assertEqual(flow_forward.shape, (1, 2, 128, 128))  # [B, 2, H, W]
        self.assertEqual(flow_backward.shape, (1, 2, 128, 128))  # [B, 2, H, W]

    def test_inference_on_dir(self):
        # Test the inference_on_dir method
        results = self.model.inference_on_dir(self.test_dir, output_path=self.output_dir)
        self.assertTrue(len(results) > 0)
        for flow, output_file in results:
            self.assertTrue(os.path.exists(output_file))
            self.assertIsInstance(flow, np.ndarray)
            self.assertEqual(flow.shape[-1], 2)  # [H, W, 2]

    def test_save_vis_flow_tofile(self):
        # Test the save_vis_flow_tofile method
        dummy_flow = np.random.rand(128, 128, 2).astype(np.float32)
        output_file = os.path.join(self.output_dir, 'test_flow.png')
        self.model.save_vis_flow_tofile(dummy_flow, output_file)
        self.assertTrue(os.path.exists(output_file))

if __name__ == '__main__':
    unittest.main()