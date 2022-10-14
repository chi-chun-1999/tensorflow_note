from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical 
import numpy as np

import matplotlib.pyplot as plt
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
def smooth_curve(points,factor = 0.9):
    smooth_points = []
    for point in points:
        if smooth_points:
            previous = smooth_points[-1]
            smooth_points.append(previous*factor+point*(1-factor))
        else:
            smooth_points.append(point)

    return smooth_points

# %%  
k = 4
num_val_samples = len(train_data) // k

num_epoch = 500

all_mae_histories = []

for i in range(k):
    print('processing fold #',i)
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_target = train_targets[i*num_val_samples:(i+1)*num_val_samples]

    partial_train_data = np.concatenate( 
            [train_data[:i*num_val_samples],
            train_data[(i+1)*num_val_samples:]],
            axis = 0
            )
    partial_train_targets = np.concatenate( 
            [train_targets[:i*num_val_samples],
            train_targets[(i+1)*num_val_samples:]],
            axis = 0
            )
    model = build_model()
    history = model.fit(partial_train_data,partial_train_targets,
            validation_data = (val_data,val_target),
           epochs=num_epoch,batch_size = 1,verbose=0)

    val_mse,val_mae = model.evaluate(val_data,val_target,verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)


# %%  
average_mae_history = [np.mean([x[i] for x in all_mae_histories])for i in range(num_epoch)]


# %%  


plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# %%  

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# %%  

model = build_model()
history = model.fit(train_data,train_targets,epochs=130,batch_size = 16,verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data,test_targets)
