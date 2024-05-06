import os

import unittest
import numpy as np

from moment.data.anomaly_detection_datasets import get_anomaly_detection_datasets, AnomalyDetectionDataset


class TestAnomalyDetectionDataset(unittest.TestCase):
    
    def setUp(self):
        self.anomaly_detection_dataset_paths = get_anomaly_detection_datasets(collection='TSB-UAD-Public')
        
    def test_anomaly_detection_dataset(self):
        for dataset_path in self.anomaly_detection_dataset_paths:
            dataset = AnomalyDetectionDataset(
                full_file_path_and_name=dataset_path,
                seq_len=512,
                data_split='val',
                scale=True,
                task='pre-training',
                train_ratio=0.6,
                val_ratio=0.1,
                test_ratio=0.3
            )

            # Add your assertions here to test the dataset
            print(f'Testing dataset: {dataset_path}')
            if dataset[0] is None:
                print(f'Error: {dataset_path} is Invalid')
                continue
            assert dataset[0] is not None
            self.assertGreater(len(dataset), 0, 'Dataset should not be empty')
            self.assertTupleEqual(dataset[0].timeseries.shape, 
                                (dataset.n_channels, dataset.seq_len), 'Dataset shape should be (n_channels, seq_len)')
            self.assertTupleEqual(dataset[0].labels.shape, 
                                (dataset.n_channels, dataset.seq_len), 'Labels shape should be (n_channels, seq_len)')
            self.assertTupleEqual(dataset[0].input_mask.shape, 
                                (dataset.seq_len,), f'Input mask shape should be (seq_len,)')
            self.assertEqual(np.isfinite(dataset.data).any(axis=None), True, 'Dataset should not have infinite/NaN values')

if __name__ == '__main__':
    unittest.main()