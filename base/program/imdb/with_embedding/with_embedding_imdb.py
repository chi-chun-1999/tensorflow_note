import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
from keras import preprocessing 
from keras.layers import Flatten, Dense, Embedding
from keras.models import Sequential
import matplotlib.pylab as plt
import numpy as np
import os

# %%  

imdb_data = pd.read_csv('/mnt/data_disk/Dataset/IMDB_Dataset.csv')

# %%  
text = imdb_data['review'].values.tolist()
label = imdb_data['sentiment'].values.tolist()
# %%  
f = lambda i:0 if i !='positive' else 1
label_transform = [ f(i) for i in label]
# %%  
maxlen = 100
trainning_samples = 200
validation_samples = 10000
max_words = 10000

# %%  
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(text)
sequence = tokenizer.texts_to_sequences(text)
# %%  
word_index = tokenizer.word_index
# %%  
data = pad_sequences(sequence,maxlen=maxlen)
labels = np.asarray(label_transform)

print(data.shape)
print(labels.shape)

# %%  
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]


# %%  

x_train = data[:trainning_samples]
y_train = labels[:trainning_samples]
x_val = data[trainning_samples:trainning_samples+validation_samples]
y_val = labels[trainning_samples:trainning_samples+validation_samples]

# %%  
glove_dir = '/mnt/data_disk/Dataset/glove.6B.100d.txt'
embedding_index = {}
f = open(glove_dir)

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    embedding_index[word] = coefs
f.close()
print(len(embedding_index))

# %%  
embedding_dim = 100

embedding_matrix = np.zeros((max_words,embedding_dim))
print(embedding_matrix.shape)

for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# %%  
model = Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=maxlen))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()
#model.layers[0].set_weights([embedding_matrix])
#model.layers[0].trainable = False
model.compile('rmsprop','binary_crossentropy',metrics=['acc'])

# %%  
history = model.fit(x_train,y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val,y_val))

# %%  
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)
# %%  

plt.clf()

plt.plot(epochs, acc, 'ro',label='Training acc')
plt.plot(epochs, val_acc, 'r',label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()

plt.show()
# %%  
plt.clf()

plt.plot(epochs, loss, 'ro',label='Training loss')
plt.plot(epochs, val_loss, 'b',label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
