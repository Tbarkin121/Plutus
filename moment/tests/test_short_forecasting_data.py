import os
import unittest

import numpy as np
import pandas as pd
from tqdm import trange

from moment.data.forecasting_datasets import \
    get_forecasting_datasets, ShortForecastingDataset


class TestShortForecastingDataset(unittest.TestCase):
    def setUp(self):
        self.forecasting_dataset_paths = get_forecasting_datasets(collection="monash")
        
    def test_forecasting_dataset(self):
        for dataset_path in self.forecasting_dataset_paths:
            dataset = ShortForecastingDataset(
                full_file_path_and_name=dataset_path,
                seq_len=512,
                pred_len=32,
                data_split='val',
                scale=True,
                task='pre-training',
                train_ratio=0.6,
                val_ratio=0.1,
                test_ratio=0.3
            )
            # Add your assertions here to test the dataset

            print(f'Testing dataset: {dataset_path}')
            assert dataset[0] is not None
            self.assertGreater(len(dataset), 0, 'Dataset should not be empty')
            self.assertEqual(dataset[0].timeseries.shape[-1], dataset.seq_len, f'Sequence length should be {dataset.seq_len}')
            self.assertEqual(dataset.n_channels, 1, 'Dataset should have 1 channel')
            self.assertTupleEqual(dataset[0].timeseries.shape, 
                                  (dataset.n_channels, dataset.seq_len), f'Dataset shape should be (n_channels, seq_len)')
            self.assertTupleEqual(dataset[0].input_mask.shape, 
                                  (dataset.seq_len,), f'Input mask shape should be (seq_len,)')
            self.assertEqual(type(dataset.data), pd.DataFrame, 'Dataset should be a pandas dataframe')
            for i in trange(dataset.data.series_value.shape[0]):
                self.assertEqual(
                    np.isfinite(dataset.data.series_value.iloc[i]).any(axis=None), 
                    True, 'Dataset should not have NaNs/Infs')
            
            # self.assertEqual(dataset[0].forecast.shape[-1], dataset.pred_len, f'Prediction length should be {dataset.pred_len}')
            # self.assertTupleEqual(dataset[0].forecast.shape, 
            #                       (dataset.n_channels, dataset.pred_len), f'Dataset shape should be (n_channels, seq_len)')

if __name__ == '__main__':
    unittest.main()