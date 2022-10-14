import tensorflow as tf

def inference(X):
    return

def loss(X,Y):
    outcome = tf.add(X,Y)
    return outcome

def inputs():
    X = tf.compat.v1.constant([5,2])
    Y = tf.compat.v1.constant([3,-6])
    return X,Y

def train(train_loss):
    learning_rate = 0.0001
    return train_loss

def evaluate(sess,X,Y):
    return


with tf.compat.v1.Session() as sess:
    #tf.global_variables_initializer()
    

    X,Y = inputs()

    total_loss = loss(X,Y)
    train_op = train(total_loss)

    c_ = sess.run([train_op])
    print('loss:',c_)




