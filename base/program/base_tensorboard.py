import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# 宣告常數A&B，後面的name參數，是要繪製tensorboard時所使用的名稱。
# 若沒有指定，或是重複名稱，則tensorboard會自動修改。
A = tf.constant(50, name='const_A')
B = tf.constant(100, name='const_B')

with tf.name_scope('Add'):
    C=A+B

with tf.compat.v1.Session() as sess:
    # 就是這邊！
    # 使用 "with tf.name_scope('Run'):" 這句話可以畫出Run這個步驟。
    with tf.name_scope('Run'):
        D=tf.constant(sess.run(C*3))
        
    
    # 畫好步驟之後，要使用"tf.summary.FileWriter"把檔案寫到目標資料夾，
    # 第二個參數表示要把整個圖層放到graph參數內，這樣才能用tensorboard畫出來。
    train_writer = tf.compat.v1.summary.FileWriter('/tmp/tensorflow_note/tfboard_Test', sess.graph)
    train_writer.close()
