from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import optimizers 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


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
network.add(layers.Dense(512,activation='relu'))
network.add(layers.Dense(1,activation='sigmoid'))

# %%
network.compile(optimizer = optimizers.RMSprop(learning_rate=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy'])

# %%
train_dir = '/mnt/data_disk/Dataset/dogs-vs-cats/small_dataset/train/'
val_dir = '/mnt/data_disk/Dataset/dogs-vs-cats/small_dataset/validation/'

train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

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
        steps_per_epoch = 100,
        epochs = 30,
        validation_data = validation_generator,
        validation_steps = 50
        )


# %%  
network.save('cats_and_dogs_small_1.h5')


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


