import unittest
import torch

from moment.utils.masking import Masking


class TestMasking(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 32
        self.patch_len = 8
        self.mask_ratio = 0.3
        self.stride = 8
        self.n_channels = 1
        self.masking_obj = Masking(
            mask_ratio=self.mask_ratio, patch_len=self.patch_len, 
            stride=self.stride)
        
    def test_convert_seq_to_patch_view(self):
        # create a random mask tensor
        mask = torch.rand(self.batch_size, self.n_channels, self.seq_len)
        mask = (mask > self.mask_ratio).long()
        n_masked_datapoints = mask.sum()
        
        # convert the mask tensor to patch view
        patch_mask = self.masking_obj.convert_seq_to_patch_view(mask, 
                                                                patch_len=self.patch_len, 
                                                                stride=self.stride)
        
        n_masked_datapoints_patch = patch_mask.sum()*self.patch_len
        
        # Check the shape of the patch mask tensor
        self.assertEqual(patch_mask.shape, (self.batch_size, 1, 4))
        self.assertGreaterEqual(n_masked_datapoints, n_masked_datapoints_patch)
    
    def test_convert_patch_to_seq_view(self):
        # Create a mask tesnor in patch view
        pass

if __name__ == '__main__':
    unittest.main()