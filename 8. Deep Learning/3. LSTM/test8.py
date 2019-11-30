from keras.models import Model
from keras.layers import Input, Dense, LSTM
import numpy as np

x = np.array([[[1.], [2.], [3.], [4.], [5.]]])
y = np.array([[6.]])

xInput = Input(batch_shape=(None, 5, 1))
xLstm_1 = LSTM(3, return_sequences=True)(xInput)
xLstm_2 = Bidirectional(LSTM(3))(xLstm_1) # many-to-many
xOutput = Dense(1)(xLstm_2)

model = Model(xInput, xOutput)
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_11 (InputLayer)        (None, 5, 1)              0         
_________________________________________________________________
lstm_12 (LSTM)               (None, 5, 3)              60        
_________________________________________________________________
bidirectional_3 (Bidirection (None, 6)                 168       
_________________________________________________________________
dense_11 (Dense)             (None, 1)                 7         
=================================================================

"""