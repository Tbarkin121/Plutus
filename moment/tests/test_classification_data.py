import os
import unittest

import numpy as np

from moment.data.classification_datasets import \
    get_classification_datasets, ClassificationDataset


class TestClassificationDataset(unittest.TestCase):
    def setUp(self):
        self.classification_dataset_paths = get_classification_datasets(collection="UCR")
        
    def test_classification_dataset(self):
        for dataset_path in self.classification_dataset_paths:
            dataset = ClassificationDataset(
                full_file_path_and_name=dataset_path,
                seq_len=512,
                data_split='train',
                scale=True,
                task='pre-training',
                train_ratio=0.6,
                val_ratio=0.1,
                test_ratio=0.3
            )
            
            print(f'Testing dataset: {dataset_path}')
            assert dataset[0] is not None
            self.assertGreater(len(dataset), 0, 'Dataset should not be empty')
            self.assertEqual(len(dataset), len(dataset.labels), 'Dataset length should match labels length')
            self.assertTupleEqual(dataset[0].timeseries.shape, 
                                    (dataset.n_channels, dataset.seq_len), 'Dataset shape should be (n_channels, seq_len)')
            self.assertTupleEqual(dataset[0].input_mask.shape, 
                                (dataset.seq_len,), f'Input mask shape should be (seq_len,)')
            self.assertEqual(np.isfinite(dataset.data).any(axis=None), True, 'Dataset should not have infinite/NaN values')
            # Add your assertions here to test the dataset
            
if __name__ == '__main__':
    unittest.main()