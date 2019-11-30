from keras.models import Model
from keras.layers import Input, Dense, LSTM
import numpy as np

x = np.array([[[1.], [2.], [3.], [4.], [5.]]])
y = np.array([[6.]])

xInput = Input(batch_shape=(None, 5, 1))
xLstm = LSTM(3)(xInput) #Lstm = LSTM(3, return_sequences=False)(xInput) many-to-one을 의미
xOutput = Dense(1)(xLstm)

model = Model(xInput, xOutput)
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

model.fit(x, y, epochs=50, batch_size=1, verbose=0)
model.predict(x, batch_size=1)

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 5, 1)              0         
_________________________________________________________________
lstm_2 (LSTM)                (None, 3)                 60        
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 4         
=================================================================

"""