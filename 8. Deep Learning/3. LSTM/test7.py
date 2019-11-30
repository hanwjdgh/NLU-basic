from keras.models import Model
from keras.layers import Input, Dense, LSTM
import numpy as np

x = np.array([[[1.], [2.], [3.], [4.], [5.]]])
y = np.array([[6.]])

xInput = Input(batch_shape=(None, 5, 1))
xLstm_1 = LSTM(3, return_sequences=True)(xInput) # many-to-one
xLstm_2 = LSTM(3)(xLstm_1)
xOutput = Dense(1)(xLstm_2)

model = Model(xInput, xOutput)
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_10 (InputLayer)        (None, 5, 1)              0         
_________________________________________________________________
lstm_10 (LSTM)               (None, 5, 3)              60        
_________________________________________________________________
lstm_11 (LSTM)               (None, 3)                 84        
_________________________________________________________________
dense_10 (Dense)             (None, 1)                 4         
=================================================================

"""