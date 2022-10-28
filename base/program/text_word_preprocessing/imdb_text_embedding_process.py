from keras.datasets import imdb
from keras import preprocessing 
from keras.layers import Flatten, Dense, Embedding
from keras.models import Sequential

# %%  


max_feature = 10000
maxlen = 20

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_feature)

# %%  

x_train = preprocessing.sequence.pad_sequences(x_train,maxlen=maxlen)

# %%  
model = Sequential()
model.add(Embedding(10000,8,input_length=20))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.compile('rmsprop','binary_crossentropy',metrics=['acc'])
model.summary()



# %%  
history = model.fit(x_train,y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)
