from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.applications import VGG16 
from tensorflow.keras import optimizers 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os
import numpy as np
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
conv_base = VGG16(weights='imagenet',
        include_top=False,
        input_shape=(150,150,3))

conv_base.trainable = False

network = models.Sequential()

network.add(conv_base)
network.add(layers.Flatten())
network.add(layers.Dense(256,activation='relu'))
network.add(layers.Dense(1,activation='sigmoid'))

# %%
network.compile(optimizer = optimizers.RMSprop(learning_rate=2e-5),
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
        batch_size = 20,
        class_mode='binary'
        )
validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(150,150),
        batch_size = 20,
        class_mode='binary'
        )

# %%  
history = network.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.samples//train_generator.batch_size,
        epochs = 30,
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

# %%  fine tune

conv_base.trainable = True

set_trainable = False 

for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


# %%  

network.compile(optimizer = optimizers.RMSprop(learning_rate=1e-5),
                loss='binary_crossentropy',
                metrics=['accuracy'])

# %%  
history = network.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.samples//train_generator.batch_size,
        epochs = 100,
        validation_data = validation_generator,
        validation_steps = validation_generator.samples//validation_generator.batch_size
        )


# %%  
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

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

# %%  

plt.plot(epochs,smooth_curve(acc),'bo',label='Trainning acc')
plt.plot(epochs,smooth_curve(val_acc),'b',label='Validation acc')
plt.title('Trainning and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,smooth_curve(loss),'bo',label='Trainning loss')
plt.plot(epochs,smooth_curve(val_loss),'b',label='Validation loss')
plt.title('Trainning and validation loss')
plt.legend()

plt.show()


# %%  
test_dir = '/mnt/data_disk/Dataset/dogs-vs-cats/small_dataset/test/'

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150,150),
        batch_size = 20,
        class_mode='binary'
        )

test_loss, test_acc = network.evaluate_generator(test_generator,steps = 50,)

print('test acc: ',test_acc)



