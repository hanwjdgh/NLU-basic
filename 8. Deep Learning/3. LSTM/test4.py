from keras.models import Model
from keras.layers import Input, Dense, LSTM, Lambda
import numpy as np

from keras import backend as K

x = np.array([[[1.], [2.], [3.], [4.], [5.]]])
y = np.array([[6.]])

xInput = Input(batch_shape=(None, 5, 1))
xLstm = LSTM(3, return_sequences=True)(xInput)
xReduced = Lambda(lambda z: K.mean(z, axis=1))(xLstm)
xOutput = Dense(1)(xReduced)

model = Model(xInput, xOutput)
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_7 (InputLayer)         (None, 5, 1)              0         
_________________________________________________________________
lstm_7 (LSTM)                (None, 5, 3)              60        
_________________________________________________________________
lambda_1 (Lambda)            (None, 3)                 0         
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 4         
=================================================================

"""