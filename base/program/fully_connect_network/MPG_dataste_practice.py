import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, Sequential
import plotext as plt

##
datset_path = tf.keras.utils.get_file("auto-mpg.data","http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

##

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year','Origin']
raw_dataset = pd.read_csv(datset_path,names=column_names,na_values ="?",comment = '\t',sep=" ",skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.head()

##

print(dataset.isna().sum())
dataset = dataset.dropna()
print(dataset.isna().sum())

## 

origin = dataset.pop('Origin')

##
dataset['USA'] = (origin ==1)*1.0
dataset['Europe'] = (origin ==2)*1.0
dataset['Japan'] = (origin ==3)*1.0
dataset.tail()

##
train_dataset = dataset.sample(frac = 0.8,random_state =0 )
test_dataset = dataset.drop(train_dataset.index)

##
train_label = train_dataset.pop('MPG')
test_label = test_dataset.pop('MPG')

##
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()


##
def norm(x):
    return (x-train_stats['mean'])/train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


##
train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values,train_label.values))
test_db = tf.data.Dataset.from_tensor_slices((normed_test_data.values,test_label.values))

train_db = train_db.shuffle(100).batch(32)

##

class Network(tf.keras.Model):
    def __init__(self):
        super(Network,self).__init__()

        self.fc1 = layers.Dense(64,activation='relu')
        self.fc2 = layers.Dense(64,activation='relu')
        self.fc3 = layers.Dense(1)

        # first layer
        #self.w1 = tf.Variable(tf.random.truncated_normal([9,64],stddev=0.1))
        #self.b1 = tf.Variable(tf.zeros([64]))

        # second layer
        #self.w2 = tf.Variable(tf.random.truncated_normal([64,64],stddev=0.1))
        #self.b2 = tf.Variable(tf.zeros([64]))
        #
        # third layer
        #self.w3 = tf.Variable(tf.random.truncated_normal([64,1],stddev=0.1))
        #self.b3 = tf.Variable(tf.zeros([1]))


    def call(self,inputs, training=None,mask=None):

        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        #h1 = inputs@self.w1 + tf.broadcast_to(self.b1,[inputs.shape[0],64])
        #h1 = tf.nn.relu(h1)

        #h2 = h1@self.w2 + self.b2
        #h2 = tf.nn.relu(h2)

        #x = h2@self.w3 + self.b3

        return x

##
#model =Network()
#
#model.build(input_shape=(4,9))
#model.summary()
#
#optimizer = tf.keras.optimizers.RMSprop(0.001)
#
#mae_statics = []
#
#
#
#
#for epoch in range(200):
#    for step, (x,y) in enumerate(train_db):
#        with tf.GradientTape() as tape:
#            out = model(x)
#            loss = tf.reduce_mean(tf.losses.MSE(y,out))
#            mae_loss = tf.reduce_mean(tf.losses.MAE(y,out))
#            mae_statics.append(mae_loss)
#
#            if epoch % 10== 0 and step % 10 ==0:
#                print('Epoch : ',epoch,'Step : ',step,'MSE : ',float(loss),'MAE: ',float(mae_loss))
#            #print(epoch,step,float(loss))
#
#            grads = tape.gradient(loss,model.trainable_variables)
#            optimizer.apply_gradients(zip(grads,model.trainable_variables))
##

valid_x = tf.constant(normed_test_data,dtype=tf.float64)
valid_y = tf.constant(test_label,dtype=tf.float64)

model =Network()

model.build(input_shape=(4,9))
model.summary()

optimizer = tf.keras.optimizers.RMSprop(0.001)

mae_statics = []
test_mae_statics = []

for epoch in range(200):
    for step, (x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = model(x)
            loss = tf.reduce_mean(tf.losses.MSE(y,out))
            mae_loss = tf.reduce_mean(tf.losses.MAE(y,out))

            test_out = model(valid_x)
            test_loss = tf.reduce_mean(tf.losses.MSE(valid_y,out))
            test_mae_loss = tf.reduce_mean(tf.losses.MAE(valid_y,out))


            if step % 10 ==0:
                print('Epoch: ',epoch,'Step: ',step,'MSE: ',float(loss),'train MAE: ',float(mae_loss),'valid MAE: ',float(test_mae_loss))
                mae_statics.append(mae_loss)
                test_mae_statics.append(test_mae_loss)
            #print(epoch,step,float(loss))

            grads = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))

##
plt.plot(mae_statics)
plt.show()
