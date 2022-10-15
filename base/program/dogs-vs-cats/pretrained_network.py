from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import optimizers 
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os
import numpy as np


# %%  
conv_base = VGG16(weights='imagenet',
        include_top=False,
        input_shape=(150,150,3))


# %%  

base_dir = r'/mnt/data_disk/Dataset/dogs-vs-cats/small_dataset'

train_dir = os.path.join(base_dir,'train')
val_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

# %%  

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory,sample_count):
    features = np.zeros(shape = (sample_count,4,4,512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
            directory,
            target_size=(150,150),
            batch_size = batch_size,
            class_mode='binary'
            )

    i = 0
    for input_batch, labels_batch in generator:
        features_batch = conv_base.predict(input_batch)
        features[i*batch_size : (i+1)*batch_size] = features_batch
        labels[i*batch_size : (i+1)*batch_size] = labels_batch
        i+=1
        print(i,end=' ')
        
        if i*batch_size >= sample_count:
            break

        return features,labels

train_features, train_labels = extract_features(train_dir,2000)
valid_features, valid_labels = extract_features(val_dir,1000)
test_features, test_labels = extract_features(test_dir,1000)

    

# %%  Let data flatten

train_features = np.reshape(train_features,(2000,4*4*512))
valid_features = np.reshape(valid_features,(1000,4*4*512))
test_features = np.reshape(test_features,(1000,4*4*512))

# %%  Build and train fully connective classifier

model = models.Sequential()
model.add(layers.Dense(256,activation='relu',input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer = optimizers.RMSprop(learning_rate=1e-5),
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = model.fit(train_features,train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(valid_features,valid_labels))

# %%  

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Trainning acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Trainning and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,loss,'bo',label='Trainning loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Trainning and validation loss')
plt.legend()

plt.show()


