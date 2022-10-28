import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras import preprocessing 
from keras.layers import Flatten, Dense, Embedding
from keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from keras import layers

# %%  
def generator(data, lookback, delay, min_index, max_index,
        shuffle=False, batch_size = 128, step = 6):
    if max_index is None:
        max_index = len(data)- delay -1

    i = max_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(
                    min_index+lookback,max_index,size=batch_size
                    )
        else:
            if i+batch_size >= max_index:
                i = min_index + lookback
            rows = np.arrange(i,min(i+batch_size,max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        
        for j,row in enumerate(rows):
            indices = range(rows[j]-lookback,rows[j],step)
            samples[j]=data[indices]
            targets[j]=data[rows[j]+delay][1]

        yield samples,targets


# %%  
def evaluate_navie_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        print(samples.shape)
        print(targets.shape)
        preds = samples[:,-1,1]
        mae = np.mean(np.abs(preds-targets))
        batch_maes.append(mae)

    print(np.mean(batch_maes))


# %%  

data_dir = '/mnt/data_disk/Dataset/jena_climate_2009_2016.csv'

data_df = pd.read_csv(data_dir)

colunm_name = list(data_df.columns)

colunm_name.pop(0)
name = colunm_name

float_data = data_df[name].to_numpy()

# %%  
plt.plot(range(len(float_data)),float_data[:,1])
plt.show()

# %%  
plt.clf()
plt.plot(range(1440),float_data[:1440,1])
plt.show()

# %%  

mean = float_data[:200000].mean(axis = 0)
float_data -= mean

std = float_data[:200000].std(axis = 0)
float_data /= std

# %%  

lookback = 1440 step = 6
delay = 144
batch_size = 128 

train_gen = generator(float_data,
                    lookback=lookback,
                    delay = delay,
                    min_index=0,
                    max_index=200000,
                    shuffle=True,
                    step=step,
                    batch_size=batch_size)

val_gen = generator(float_data,
                    lookback=lookback,
                    delay = delay,
                    min_index=200001,
                    max_index=300000,
                    shuffle=True,
                    step=step,
                    batch_size=batch_size)

test_gen = generator(float_data,
                    lookback=lookback,
                    delay = delay,
                    min_index=300001,
                    max_index=None,
                    shuffle=True,
                    step=step,
                    batch_size=batch_size)

val_steps = (300000-200001-lookback)//batch_size
test_steps = (len(float_data)-300001-lookback)//batch_size

# %%  
evaluate_navie_method()

# %% Only using Dense Layer
model = Sequential()
model.add(layers.Flatten(input_shape=(lookback//step,float_data.shape[-1])))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(1))




# %%  
model.compile(optimizer=RMSprop(),loss='mae')
history = model.fit_generator(train_gen,
                            steps_per_epoch = 500,
                            epochs=20,
                            validation_data=val_gen,
                            validation_steps = val_steps)

# %%  
model = Sequential()
model.add(layers.GRU(32,input_shape=(None,float_data.shape[-1])))
model.add(layers.Dense(1))
model.summary()




# %%  
model.compile(optimizer=RMSprop(),loss='mae')
history = model.fit_generator(train_gen,
                            steps_per_epoch = 500,
                            epochs=20,
                            validation_data=val_gen,
                            validation_steps = val_steps)

# %%  
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.clf()

plt.plot(epochs, loss, 'ro',label='Training loss')
plt.plot(epochs, val_loss, 'r',label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()

# %%  

model = Sequential()
model.add(layers.GRU(32,dropout=0.2,recurrent_dropout=0.2,input_shape=(None,float_data.shape[-1])))
model.add(layers.Dense(1))
model.summary()




# %%  
model.compile(optimizer=RMSprop(),loss='mae')
history = model.fit_generator(train_gen,
                            steps_per_epoch = 500,
                            epochs=40,
                            validation_data=val_gen,
                            validation_steps = val_steps)

# %%  
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.clf()

plt.plot(epochs, loss, 'ro',label='Training loss')
plt.plot(epochs, val_loss, 'r',label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
