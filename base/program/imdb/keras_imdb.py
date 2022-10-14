import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras import models
from tensorflow.keras import layers

import matplotlib.pyplot as plt

# %%
def vecotrize_sequencs(sequences, dimesion=10000):
    results = np.zeros((len(sequences),dimesion))

    for i, sequences in enumerate(sequences):
        results[i,sequences] = 1.

    return results

# %%

(train_data,train_label),(test_data,test_label) = imdb.load_data(num_words=10000)

# %%
train_x = vecotrize_sequencs(train_data)
test_x = vecotrize_sequencs(test_data)

train_y = np.array(train_label).astype('float32')
test_y = np.array(test_label).astype('float32')

# %%
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(17,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

# %%  

model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])


# %%  
x_val = train_x[:10000]
partial_x_val = train_x[10000:]


y_val = train_y[:10000]
partial_y_val = train_y[10000:]



# %%  

history = model.fit(partial_x_val,
                    partial_y_val,
                    epochs=4,
                    batch_size=512,
                    validation_data=(x_val,y_val)
                    )


# %% 

history_dict = history.history
loss_value = history_dict['loss']
val_loss_value = history_dict['val_loss']

epochs = range(1, len(loss_value)+1)

plt.plot(epochs, loss_value, 'ro',label='Training loss')
plt.plot(epochs, val_loss_value, 'b',label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# %%  

plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.plot(epochs, acc, 'ro',label='Training acc')
plt.plot(epochs, val_acc, 'r',label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()

plt.show()


# %%  

evaluate_value = model.evaluate(test_x,test_y)
print(evaluate_value)


# %%  

model.predict(test_x)
