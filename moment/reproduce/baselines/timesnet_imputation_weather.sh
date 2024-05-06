### Weather
python3 ../../scripts/baselines/timesnet_imputation.py\
 --config '../../configs/imputation/timesnet_train.yaml'\
 --gpu_id 2\
 --d_model 64\
 --d_ff 64\
 --n_channels 21\
 --train_batch_size 32\
 --val_batch_size 128\
 --dataset_names '/XXXX-14/project/public/XXXX-9/TimeseriesDatasets/forecasting/autoformer/weather.csv'
