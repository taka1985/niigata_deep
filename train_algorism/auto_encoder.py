import tensorflow as tf

class AutoEncoder:
    @classmethod
    def Encoding(cls,input,num_node):

        def auto_encoding(x, w_enc, b_enc, w_dec, b_dec):
            encoded = tf.sigmoid(tf.matmul(x, w_enc) + b_enc)
            decoded = tf.matmul(encoded, w_dec) + b_dec
            return encoded,decoded

        len,num = input.shape
        if not type(num) == int:
            num = int(num)
        x = tf.placeholder(tf.float32,[None,num])
        w_enc = tf.Variable(tf.random_normal([num,num_node],stddev = 0.05))
        b_enc = tf.Variable(tf.zeros([num_node]))

        w_dec = tf.Variable(tf.random_normal([num_node, num], stddev=0.05))
        b_dec = tf.Variable(tf.zeros([num]))

        #初回の自己符号下記の誤差関数
        encoded,decoded = auto_encoding(x, w_enc, b_enc, w_dec, b_dec)
        rmse = tf.sqrt(
            tf.reduce_mean(tf.square(tf.subtract(x, decoded))))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(rmse)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        i = 0
        for _ in range(300):
            i += 1
            sess.run(train_step,feed_dict={x:input})
            if i % 100 == 0:
                print("epoch_encode:{0}".format(i))
        w_dec_p,b_dec_p,w_enc_p,b_enc_p = sess.run([w_dec, b_dec, w_enc, b_enc])
        sess.close()
        return w_enc_p,b_enc_p