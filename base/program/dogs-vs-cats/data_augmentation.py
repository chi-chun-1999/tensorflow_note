from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import optimizers 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os
import numpy as np

# %%

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )


# %%  
train_cats_dir = '/mnt/data_disk/Dataset/dogs-vs-cats/small_dataset/train/cats'

fnames = [os.path.join(train_cats_dir,fname) for fname in os.listdir(train_cats_dir)]

img_path = fnames[3]
img = image.load_img(img_path,target_size=(150,150))

x = image.img_to_array(img)
plt.show()
x = x.reshape((1,)+x.shape)

# %%  

i = 0
for batch in datagen.flow(x,batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
# %%

network = models.Sequential()

network.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(64,(3,3),activation='relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(128,(3,3),activation='relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(128,(3,3),activation='relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Flatten())
network.add(layers.Dropout(0.5))
network.add(layers.Dense(512,activation='relu'))
network.add(layers.Dense(1,activation='sigmoid'))

# %%
network.compile(optimizer = optimizers.RMSprop(learning_rate=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy'])

# %%  

train_dir = '/mnt/data_disk/Dataset/dogs-vs-cats/small_dataset/train/'
val_dir = '/mnt/data_disk/Dataset/dogs-vs-cats/small_dataset/validation/'

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        )
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size = 32,
        class_mode='binary'
        )
validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(150,150),
        batch_size = 32,
        class_mode='binary'
        )

# %%  
history = network.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.samples//train_generator.batch_size,
        epochs = 10,
        validation_data = validation_generator,
        validation_steps = validation_generator.samples//validation_generator.batch_size
        )

# %%  
network.save('cats_and_dogs_small_2.h5')

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
