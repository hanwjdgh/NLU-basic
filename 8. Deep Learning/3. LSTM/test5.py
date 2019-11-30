from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional
import numpy as np

x = np.array([[[1.], [2.], [3.], [4.], [5.]]])
y = np.array([[6.]])

xInput = Input(batch_shape=(None, 5, 1))
xBiLstm = Bidirectional(LSTM(3), merge_mode='concat')(xInput) # many-to-one
xOutput = Dense(1)(xBiLstm)

model = Model(xInput, xOutput)
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_8 (InputLayer)         (None, 5, 1)              0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 6)                 120       
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 7         
=================================================================

"""