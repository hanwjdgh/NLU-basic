from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(3, input_dim=4, activation='softmax'))
