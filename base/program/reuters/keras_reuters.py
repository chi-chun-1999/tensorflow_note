import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import reuters
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
def to_one_hot(label,dimesion=46):
    result = np.zeros((len(label),dimesion))
    for i,label in enumerate(label):
        result[i,label] = 1.

    return result
# %%

(train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words=10000) 
# %%

word_index = reuters.get_word_index()
reverse_word_index = dict([(num,name) for (name,num) in word_index.items()])
# %% 

x_train = vecotrize_sequencs(train_data)
x_test = vecotrize_sequencs(test_data)

# %%  
one_hot_train_label = to_one_hot(train_labels)
one_hot_test_label = to_one_hot(test_labels)


# %%
model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(4,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

# %%  

model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# %%  
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_label[:1000]
partial_y_train = one_hot_train_label[1000:]
# %%  
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 10,
                    batch_size = 512,
                    validation_data = (x_val,y_val))

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

evaluate_value = model.evaluate(x_test,one_hot_test_label)
print(evaluate_value)
# %%  

prediction  = model.predict(x_test)


