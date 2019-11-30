from keras.models import Model
from keras.layers import Input, Dense, LSTM, TimeDistributed
import numpy as np

x = np.array([[[1.], [2.], [3.], [4.], [5.]]])
y = np.array([[[2.], [3.], [4.], [5.], [6.]]])

xInput = Input(batch_shape=(None, 5, 1))
xLstm = LSTM(3, return_sequences=True)(xInput) # return_sequence=True & TimeDistributed() = many-to-many
xOutput = TimeDistributed(Dense(1))(xLstm)

model = Model(xInput, xOutput)
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_4 (InputLayer)         (None, 5, 1)              0         
_________________________________________________________________
lstm_4 (LSTM)                (None, 5, 3)              60        
_________________________________________________________________
time_distributed_2 (TimeDist (None, 5, 1)              4         
=================================================================

"""