# Plutus
Stock Forecasting

data : firstratedata.com/free-intraday-data


Thoughts : 

LSTM (Long Short-Term Memory) Networks: These are a type of recurrent neural network (RNN) suitable for sequence prediction problems. LSTMs can remember information for a long period of time, making them ideal for time series forecasting like stock prices.

GRU (Gated Recurrent Units): GRUs are similar to LSTMs but are simpler and can be more efficient to compute and train. They also perform well on sequence prediction tasks and can be a good choice depending on the complexity of your problem.

1D Convolutional Neural Networks: While traditionally used for image processing, 1D CNNs can be applied to time series data for capturing temporal dependencies. They can be faster to train than RNNs and are effective in extracting features from sequences.

Transformer Models: These have been very successful in natural language processing and are also being adapted for time series forecasting. Transformers use self-attention mechanisms to weigh the importance of different points in the input data.

Hybrid Models: Combining different types of networks, such as CNNs for feature extraction followed by LSTMs or GRUs to model the time series data, can leverage the strengths of each model type.


Current State

Implimented an LSTM. Long sequences have issues training correctly. (Vanishing Gradients?) 

Things to try : 
1) Gradient Clipping : Clip the gradients... 

2) Sequence Bucketing : sequence bucketing is where sequences are divided into buckets of similar lengths (e.g., 100-200 timesteps), and each bucket is padded to the maximum length within the bucket. This way, the LSTM doesn’t always have to handle the maximum sequence length, reducing computational overhead and improving learning dynamics.

3) Switching to Attention Mechanisms and Transformers 

