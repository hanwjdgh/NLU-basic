from keras.models import Model
from keras.layers import Input, Dense, LSTM
from keras.layers import Bidirectional, TimeDistributed
import numpy as np

x = np.array([[[1.], [2.], [3.], [4.], [5.]]])
y = np.array([[[2.], [3.], [4.], [5.], [6.]]])

xInput = Input(batch_shape=(None, 5, 1))
xBiLstm = Bidirectional(LSTM(3, return_sequences=True),merge_mode='concat')(xInput) # many-to-many
xOutput = TimeDistributed(Dense(1))(xBiLstm)

model = Model(xInput, xOutput)
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

"""
Layer (type)                 Output Shape              Param #   
=================================================================
input_9 (InputLayer)         (None, 5, 1)              0         
_________________________________________________________________
bidirectional_2 (Bidirection (None, 5, 6)              120       
_________________________________________________________________
time_distributed_4 (TimeDist (None, 5, 1)              7         
=================================================================

"""