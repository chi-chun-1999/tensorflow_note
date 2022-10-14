from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical 
import numpy as np
# %%

(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data()

# %%  

mean = train_data.mean(axis=0)
train_data-=mean
std = train_data.std(axis = 0)
train_data/=std

test_data -= mean
test_data /= std


# %%  

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',
                            input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1,activation='relu'))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

# %%  


