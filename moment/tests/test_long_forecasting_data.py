import os
import unittest
import warnings

import numpy as np

from moment.data.forecasting_datasets import \
    get_forecasting_datasets, LongForecastingDataset


class TestLongForecastingDataset(unittest.TestCase):
    def setUp(self):
        self.forecasting_dataset_paths = get_forecasting_datasets(collection="autoformer")
        
    def test_forecasting_dataset(self):
        for dataset_path in self.forecasting_dataset_paths:
            with warnings.catch_warnings(record=True) as w:
                dataset = LongForecastingDataset(
                    full_file_path_and_name=dataset_path,
                    seq_len=512,
                    forecast_horizon=32,
                    data_split='train',
                    scale=True,
                    task='pre-training',
                    train_ratio=0.6,
                    val_ratio=0.2,
                    test_ratio=0.2
                )
                
                if dataset.length_timeseries < dataset.seq_len:
                    print(f'Skip dataset: {dataset_path}')
                    continue
                print(f'Testing dataset: {dataset_path}')
                assert dataset[0] is not None
                self.assertGreater(len(dataset), 0, 'Dataset should not be empty')
                self.assertTupleEqual(dataset[0].timeseries.shape, 
                                    (dataset.n_channels, dataset.seq_len), 'Dataset shape should be (n_channels, seq_len)')
                self.assertTupleEqual(dataset[0].forecast.shape, 
                                    (dataset.n_channels, dataset.forecast_horizon), 'Dataset shape should be (n_channels, pred_len)')
                self.assertTupleEqual(dataset[0].input_mask.shape, 
                                  (dataset.seq_len,), f'Input mask shape should be (seq_len,)')
                self.assertEqual(np.isfinite(dataset.data).any(axis=None), True, 'Dataset should not have infinite/NaN values')
                # Add your assertions here to test the dataset

                # check for warnings
                if len(w) > 0:
                    print(f'Warnings for dataset: {dataset_path}')
                    for warning in w:
                        print(f'Warning message: {warning.message}')
                
if __name__ == '__main__':
    unittest.main()