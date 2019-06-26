import numpy as np
import tensorflow as tf
from makedata.makedataset import makedataset
from train_algorism.auto_encoder import AutoEncoder

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def rule(month):
    if month == 12 or month == 1 or month == 2:
        return False
    else:
        return True

#データセットの準備
dbname = "makedata\Kamedago_DataSet.db"

DataMaker = makedataset(dbname,10,1)
DataMaker(rule,"Event","DataBox","Inflow.csv")
DataMaker(rule,"TestEvent","DataBox_Test","Inflow2018.csv")

x_train,t_train,x_test,t_test = DataMaker.getdata(testyear=2018)

#自己符号化器の宣言
AutoEncoder = AutoEncoder()
auto_encoding = False #事前学習の有無

#入力値の長さ
input_length = 60
#隠し層のノード数{H1:第1層,H2:第2層}
H1 = 60
H2 = 30
#学習の条件を指定
epoch_num = 500
early_stop = False
num_data = len(x_train)
batch_size = 512

x = tf.placeholder(tf.float32, [None, input_length])
t = tf.placeholder(tf.float32, [None, 1])

#ネットワークの構築
w1 = tf.Variable(tf.random_normal([input_length,H1]))
b1 = tf.zeros(H1)
w2 = tf.Variable(tf.random_normal([H1,H2]))
b2 = tf.zeros(H2)
w3 = tf.Variable(tf.random_normal([H2,1]))
b3 = tf.zeros(1)

a1 = tf.sigmoid(tf.matmul(x,w1) + b1)
a2 = tf.sigmoid(tf.matmul(a1,w2) + b2)
y = tf.matmul(a2,w3) + b3

cost = tf.reduce_mean(tf.square(y - t))
train = tf.train.AdamOptimizer(1e-4).minimize(cost)

with tf.Session() as sess:
    if auto_encoding:
        w_enc_p,b_enc_p = AutoEncoder(x_train_enc,H1,sess)
        w1 = tf.Variable(w_enc_p)
        b1 = tf.Variable(b_enc_p)
        sec_x = sigmoid(np.matmul(x_train_enc,np.array(w_enc_p)) + np.array(b_enc_p))
        w2_enc_p, b2_enc_p = AutoEncoder.Encoding(sec_x, H2)
        w2 = tf.Variable(w2_enc_p)
        b2 = tf.Variable(b2_enc_p)

        i = 0
        while i <= epoch_num and early_stop == False:
            i += 1

            # ミニバッチ学習
            sff_idx = np.random.permutation(num_data)
            for idx in range(0, num_data, batch_size):
                x_batch = x_train[sff_idx[idx: idx + batch_size if idx + batch_size < num_data else num_data]]
                y_batch = t_train[sff_idx[idx: idx + batch_size if idx + batch_size < num_data else num_data]]
                _,now_cost = sess.run([train,cost], feed_dict={x: x_batch, t: y_batch})

