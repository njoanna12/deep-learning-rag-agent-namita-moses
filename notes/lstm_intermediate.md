\# LSTM Networks



\## Overview



A Long Short-Term Memory network (LSTM) is a type of recurrent neural network designed to learn long-range dependencies in sequences. It was created to fix the vanishing gradient problem that makes ordinary RNNs forget information from many steps earlier.



\## The Cell State



The cell state is a running memory that passes through the network with only small, controlled changes. Because information can flow along it almost unchanged, gradients do not shrink to nothing during training, which lets the network remember things over long sequences.



\## The Gates



An LSTM uses three gates to control the cell state. The forget gate decides what information to remove from the cell state. The input gate decides what new information to add. The output gate decides what part of the cell state to expose as the hidden state at the current step. Each gate is a small neural layer with a sigmoid activation that outputs values between 0 and 1, where 0 means block everything and 1 means let everything through.



\## Common Uses



LSTMs are used for language modeling, machine translation, speech recognition, and time series forecasting. They have been largely replaced by transformers for many language tasks, but they are still useful for smaller sequence problems.

