'''
Created on 2018年10月12日

@author: hxy
'''
'''
Created on 2018年10月2日

@author: hxy
'''
import time
import tensorflow as tf
import pickle
from tensorflow.contrib import rnn
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np


'''加载数据'''
with open("./data/test_with_length.pkl", 'rb') as inp:
    X_test = pickle.load(inp)
    y_test = pickle.load(inp)
    l_test = pickle.load(inp)
    inp.close()

'''设置对应的训练参数'''
learning_rate =  1e-3
epoch = 100
max_samples = 11942
batch_size = 50
display_step = 238
layer_num = 1
n_input = 200
n_steps = 50
n_hidden = 256
n_classes = 3
keep_prob = 1.0
max_grad_norm = 15
attn_length = 5

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.int32, [None, n_steps])
l = tf.placeholder(tf.int32, [None])
weights = tf.Variable(tf.random_normal([n_hidden * 2, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

def single_np(arr, target):
    arr = np.array(arr)
    mask = (arr == target)
    arr_new = arr[mask]
    return arr_new.size

'''构建Graph'''


def GRU_cell():
    cell = rnn.GRUCell(n_hidden, reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)


inputs = tf.transpose(x, [1, 0, 2])
inputs = tf.reshape(inputs, [-1, n_input])
inputs = tf.split(inputs, n_steps)
# ** 1.构建前向后向多层 LSTM
cell_fw = rnn.MultiRNNCell([GRU_cell() for _ in range(layer_num)], state_is_tuple=True)
cell_bw = rnn.MultiRNNCell([GRU_cell() for _ in range(layer_num)], state_is_tuple=True)
# ** 2.初始状态+
initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
# ** 3.bi-lstm 计算（tf封装）
outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs,
                                             initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
                                             dtype=tf.float32, sequence_length=l)
output = tf.reshape(tf.concat(outputs, 1), [-1, 2 * n_hidden])
logits = tf.matmul(output, weights) + biases

tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)
regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
reg_term = tf.contrib.layers.apply_regularization(regularizer)

correct_prediction = tf.equal(tf.cast(tf.argmax(tf.nn.softmax(logits), 1), tf.int32), tf.reshape(y, [-1]))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y, [-1]), logits=logits)+reg_term)
#cost = tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg_term)
'''***** 优化求解 *******'''
tvars = tf.trainable_variables()  # 获取模型的所有参数
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)  # 获取损失函数对于每个参数的梯度
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # 优化器
train_op = optimizer.apply_gradients(zip(grads, tvars),
                                     global_step=tf.contrib.framework.get_or_create_global_step())  # 梯度下降计算

saver = tf.train.Saver(max_to_keep=1)
init = tf.global_variables_initializer()
init2 = tf.local_variables_initializer()
with tf.Session() as sess:
    target_names = ['负样本', '正样本', "填充值"]
    result_df = pd.DataFrame(columns=('code行数', '正样本pre', '正样本rec', '正样本f1', '正样本count', '负样本pre','负样本rec','负样本f1','负样本count','amount'))
    saver.restore(sess,"./model/testmodel3.ckpt")
    for comment_iter in range(1,16):
        count = 0
        pre_list = []
        true_list = []
        for i in range(len(X_test)):
            data = np.array(X_test[i])
            data = np.tile(data, (1, batch_size))
            data = data.reshape((batch_size, n_steps, n_input))
            label = y_test[i]
            length = np.tile(l_test[i], batch_size)
            comment_num = single_np(label, 1)
            if comment_num == comment_iter:
                count += 1
               # print(i)
                feed_dict = {x: data, y: label, l:length}
                result = sess.run(logits,feed_dict)
                y_pre = tf.argmax(sess.run(logits,feed_dict),1).eval()
                y_pred = y_pre[0:50]
                for item in y_pred:
                    pre_list.append(item)
                for item2 in label:
                    true_list.append(item2)
        true_all = np.array(true_list).reshape(-1,1)
        pre_all = np.array(pre_list)
        print("comment length="+str(comment_iter)+"       "+str(count))
        if count != 0:
            print(classification_report(true_all, pre_all, target_names=target_names, digits=3))
            result_dict = classification_report(true_all, pre_all, target_names=target_names, digits=3, output_dict=True)
            label1_dict = result_dict['负样本']
            label2_dict = result_dict['正样本']
            label3_dict = result_dict["填充值"]
            result_df = result_df.append(pd.DataFrame({'comment数':[comment_iter],'正样本pre':[label2_dict['precision']],'正样本rec':[label2_dict['recall']],'正样本f1':[label2_dict['f1-score']],'正样本count':[label2_dict['support']],
                                                      '负样本pre':[label1_dict['precision']],'负样本rec':[label1_dict['recall']],'负样本f1':[label1_dict['f1-score']],'负样本count':[label1_dict['support']],'amount':[count]}),ignore_index=True)

    result_df.to_csv("./testresult/testresult_commentlen.csv", encoding='gbk', index = False)