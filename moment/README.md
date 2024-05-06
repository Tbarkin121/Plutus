# MOMENT
Official code for the paper MOMENT: A Family of Open Time-series Foundation Models. 

## Introduction
We introduce MOMENT, a family of open-source foundation models for general-purpose time-series analysis. Pre-training large models on time-series data is challenging due to (1) the absence a large and cohesive public time-series repository, and (2) diverse time-series characteristics which make multi-dataset training onerous. Additionally, (3) experimental benchmarks to evaluate these models especially in scenarios with limited resources, time, and supervision, are still in its nascent stages. To address these challenges, we compile a large and diverse collection of public time-series, called the Time-series Pile, and systematically tackle time-series-specific challenges to unlock large-scale multi-dataset pre-training. Finally, we build on recent work to design a benchmark to evaluate time-series foundation models on diverse tasks and datasets in limited supervision settings. Experiments on this benchmark demonstrate the effectiveness of our pre-trained models with minimal data and task-specific fine-tuning. Finally, we present several interesting empirical observations about large pre-trained time-series models.

## Usage

Install the package using:
```bash
pip install git+XXXX
```

To use the model, you can use the following code:
```python
from models.moment import MOMENTPipeline

# Options: "pre-training", "short-horizon-forecasting", "long-horizon-forecasting", "classification", "imputation", "anomaly-detection", "embed"
task_name = "classification"  

model = BETTPipeline.from_pretrained(
    "AutonLab/test-t5-small",
    model_kwargs={
        "task_name": task_name,
        "n_channels": 1,
        "num_class": 2,
    },
)
model.init()
```

## Installation

Required Python version: 3.11.5

To install the required packages, run the following command in the directory with the `setup.py` file:

```bash
> # Create a Conda environment
> conda create -n moment python=3.11.5
> # Activate the environment
> conda activate moment 
> # Install all the dependencies
> pip install git+XXXX
```


## Pre-training

To start the pre-training procedure, run the following command:
```bash
python3 -m scripts.pretraining --config configs/default.yaml --gpu_id 0
```

## Experiments Reproduction

First create a `.env` file in the `MOMENT/` directory, and add the following environment paths: 

```bash
## Huggingface Cache Directories
HF_HOME=SCRATCH_PATH/.cache/huggingface

## MOMENT project Environment Variables
MOMENT_DATA_DIR=/XXXX-14/project/public/XXXX-9/TimeseriesDatasets
MOMENT_CHECKPOINTS_DIR=SCRATCH_PATH/moment_checkpoints
MOMENT_RESULTS_DIR=SCRATCH_PATH/moment_results

# Weights and Biases Environment Variables
WANDB_DIR=SCRATCH_PATH/wandb/wandb
WANDB_CACHE_DIR=SCRATCH_PATH/.cache/wandb
```

To reproduce results from the paper, run one of the scripts from the root directory:
```bash
bash reproduce/FPT/MOMENT.sh
```

<a id="contribution"></a>
## Contributions
We encourage researchers to contribute their methods and datasets to MOMENT. We are actively working on contributing guidelines. Stay tuned for updates!

<a id="license"></a>
## License

MIT License

Copyright (c) 2024 XXXX-10 Lab, XXXX-12

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See [MIT LICENSE](XXXX) for details.
