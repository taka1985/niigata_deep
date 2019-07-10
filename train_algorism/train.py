import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from makedata.makedataset import makedataset


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rule(month):
    if month == 12 or month == 1 or month == 2:
        return False
    else:
        return True

class Dense:
    def __init__(self,input,node_num,w_enc=None,b_enc=None):
        if w_enc is not None and b_enc is not None:
            self.w = tf.Variable(w_enc)
            self.b = tf.Variable(b_enc)
        else:
            self.w = tf.Variable(tf.float32, [input, node_num])
            self.b = tf.zeros(node_num)

    def __call__(self,h):
        return tf.nn.sigmoid(tf.matmul(h, self.w) + self.b)

class train:
    @classmethod
    def go(cls,filenum,testyear):
        from train_algorism.auto_encoder import AutoEncoder

        # データセットの準備
        dbname = "makedata\Kamedago_DataSet.db"
        dataholder = "datasets\dataset.db"

        DataMaker = makedataset(dbname, 10, 1)
        already_exist = DataMaker(rule, "Event", "DataBox", "Inflow.csv")
        if not already_exist:
            DataMaker(rule, "TestEvent", "DataBox_Test", "Inflow2018.csv",True)

        x_train, t_train, x_test, t_test,testday = DataMaker.getdata(testyear=2017)
        print("Dataset is ready")
        # 自己符号化器の宣言
        AutoEncoder = AutoEncoder()
        auto_encoding = True  # 事前学習の有無

        # 入力値の長さ
        input_length = len(x_train[0])
        # 隠し層のノード数{H1:第1層,H2:第2層}
        H1 = 60
        H2 = 30
        layers_node = [H1,H2]
        # 学習の条件を指定
        epoch_num = 300
        early_stop = False
        num_data = len(x_train)
        batch_size = 512

        x = tf.placeholder(tf.float32, [None, input_length])
        t = tf.placeholder(tf.float32, [None, 1])

        # ネットワークの構築
        layers = []
        in_h = input_length
        input_for_enc = x_train
        if auto_encoding:
            for h in layers_node:
                w_enc,b_enc = AutoEncoder.Encoding(input_for_enc,h)
                layers.append(Dense(in_h,h,w_enc,b_enc))
                input_for_enc = sigmoid(np.matmul(input_for_enc, np.array(w_enc)) + np.array(b_enc))
                in_h = h
            # w_enc_p, b_enc_p = AutoEncoder.Encoding(x_train, H1)
            # w1 = tf.Variable(w_enc_p)
            # b1 = tf.Variable(b_enc_p)
            # sec_x = sigmoid(np.matmul(x_train, np.array(w_enc_p)) + np.array(b_enc_p))
            # w2_enc_p, b2_enc_p = AutoEncoder.Encoding(sec_x, H2)
            # w2 = tf.Variable(w2_enc_p)
            # b2 = tf.Variable(b2_enc_p)
        else:
            # w1 = tf.Variable(tf.random_normal([input_length, H1]))
            # b1 = tf.Variable(H1)
            # w2 = tf.Variable(tf.random_normal([H1, H2]))
            # b2 = tf.Variable(H2)
            for h in layers_node:
                layers.append(Dense(in_h,h))
                in_h = h


        w3 = tf.Variable(tf.random_normal([layers_node[-1], 1]))
        b3 = tf.zeros(1)

        h = x
        for layer in layers:
            h = layer(h)
        # h1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
        # h2 = tf.nn.sigmoid(tf.matmul(h1, w2) + b2)
        y = tf.matmul(h, w3) + b3

        cost = tf.reduce_mean(tf.square(y - t))
        train = tf.train.AdamOptimizer(1e-4).minimize(cost)

        initialize = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(initialize)
            i = 0
            for epoch in range(epoch_num):
                # ミニバッチ学習
                sff_idx = np.random.permutation(num_data)
                for idx in range(0, num_data, batch_size):
                    x_batch = x_train[sff_idx[idx: idx + batch_size if idx + batch_size < num_data else num_data]]
                    y_batch = t_train[sff_idx[idx: idx + batch_size if idx + batch_size < num_data else num_data]]
                    _, now_cost = sess.run([train, cost], feed_dict={x: x_batch, t: y_batch})

                if (epoch + 1) % 100 == 0:
                    valid_cost = sess.run(cost, feed_dict={x: x_test, t: t_test})
                    print("EPOCH: %i, TrainingCost: %.3f,TestCost: %.3f" % (epoch + 1, now_cost, valid_cost))

            y_estim,test_loss = sess.run([y,cost], feed_dict={x: x_test,t:t_test})


        path = "result"
        np.savetxt(os.path.join(path,"d{0}_{1}.csv".format(testyear,filenum)),testday,delimiter=',',fmt='%s')
        np.savetxt(os.path.join(path,"r{0}_{1}.csv".format(testyear,filenum)),t_test,delimiter=',',fmt='%s')
        np.savetxt(os.path.join(path,"e{0}_{1}.csv".format(testyear,filenum)),y_estim,delimiter=',',fmt='%s')

        return test_loss
        # plt.plot(y_estim,label = "estimate",color = "orange")
        # plt.plot(t_test,label = "real",color = "blue")
        # plt.legend()
        # plt.show()
