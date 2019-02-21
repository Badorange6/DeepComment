'''
Created on 2018年10月12日

@author: hxy
'''
'''
Created on 2018年10月2日

@author: hxy
'''
import numpy as np
import sys
import tensorflow as tf
import pickle
from tensorflow.contrib import rnn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

def split_train(data, total, splitsize):
    np.random.seed(8)
    shuffled_indices = np.random.permutation(total)
    test_set_size = int(splitsize)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[test_indices], data[train_indices]


file1 = open("./testresult/testresult.txt", 'w', encoding='utf-8')
'''加载数据'''
with open("./data/train_with_length.pkl", 'rb') as inp:
    X_train = pickle.load(inp)
    y_train = pickle.load(inp)
    l_train = pickle.load(inp)
    inp.close()

with open("./data/valid_with_length.pkl", 'rb') as inp:
    X_valid = pickle.load(inp)
    y_valid = pickle.load(inp)
    l_valid = pickle.load(inp)
    inp.close()

with open("./data/test_with_length.pkl", 'rb') as inp:
    X_test = pickle.load(inp)
    y_test = pickle.load(inp)
    l_test = pickle.load(inp)
    inp.close()


'''设置对应的训练参数'''
learning_rate = 2e-3
epoch = 60
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


x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.int32, [None, n_steps])
l = tf.placeholder(tf.int32, [None])
weights = tf.Variable(tf.random_normal([n_hidden * 2, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))


def test_epoch(test_data, test_label, test_len):
    """Testing or valid."""
    data_size = test_label.shape[0]
    batch_num = int(data_size / batch_size)
    _costs = 0.0
    _accs = 0.0
    _recalls = 0.0
    step = 1
    target_names = ['负样本', '正样本', '填充值']
    pre_list = []
    true_list = []
    for i in range(batch_num):
        X_batch = np.array(test_data[(step - 1) * batch_size: step * batch_size])
        y_batch = np.array(test_label[(step - 1) * batch_size: step * batch_size])
        l_batch = np.array(test_len[(step - 1) * batch_size: step * batch_size])
        X_batch = X_batch.reshape((batch_size, n_steps, n_input))
        y_batch = y_batch.reshape((-1, n_steps))
        l_batch = l_batch.reshape(-1)
        feed_dict = {x: X_batch, y: y_batch, l: l_batch}
        batch_acc = sess.run(accuracy, feed_dict)
        batch_cost = sess.run(cost, feed_dict)
        _costs += batch_cost
        _accs += batch_acc
        y_pre = tf.argmax(sess.run(logits, feed_dict), 1)
        y_true = y_batch.reshape([-1])
        for item in y_pre.eval():
            pre_list.append(item)
        for item2 in y_true:
            true_list.append(item2)
        step += 1
    true_all = np.array(true_list)
    pre_all = np.array(pre_list)
    result_dict = classification_report(true_all, pre_all, target_names=target_names, digits=3, output_dict=True)
    label2_dict = result_dict["正样本"]
    label2_f1 = label2_dict["f1-score"]
    label2_rec = label2_dict['recall']
    file1.write("###################################################################\n")
    file1.write(classification_report(true_all, pre_all, target_names=target_names, digits=3))
    file1.write("valid_acc =" + str(_accs / (step - 1)) + "      valid_cost =" + str(_costs / (step - 1)) + "\n")
    file1.flush()
    return label2_f1, label2_rec


def epoch_shuffle(train_data, train_label, train_len):
    permutation = np.random.permutation(train_label.shape[0])
    shuffled_dataset = train_data[permutation, :]
    shuffled_labels = train_label[permutation, :]
    shuffled_len = train_len[permutation, :]
    return shuffled_dataset, shuffled_labels, shuffled_len


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
# ** 3.bi-lstm 计算
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
    sess.run(init)
    sess.run(init2)
    #saver.restore(sess, "./model/testmodel.ckpt")
    epoch_num = 0
    g_step = 0
    f1_tmp = 0
    while epoch_num < epoch:
        step = 1
        accs = 0
        losses = 0
        break_flag = 0
        while step * batch_size <= max_samples:
            batch_x = np.array(X_train[(step - 1) * batch_size: step * batch_size])
            batch_y = np.array(y_train[(step - 1) * batch_size: step * batch_size])
            batch_l = np.array(l_train[(step - 1) * batch_size: step * batch_size])
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            batch_y = batch_y.reshape((-1, n_steps))
            batch_l = batch_l.reshape(-1)
            fetches = [accuracy, cost, train_op]
            feed_dict = {x: batch_x, y: batch_y, l: batch_l}
            sess.run(fetches, feed_dict)
            pre = tf.argmax(sess.run(logits, feed_dict), 1)
            acc = sess.run(accuracy, feed_dict)
            loss = sess.run(cost, feed_dict)
            accs += acc
            losses += loss
            if step % display_step == 0:
                _acc = accs / display_step
                _loss = losses / display_step
                accs = 0
                losses = 0
                print("Iter" + str(step * batch_size + epoch_num * max_samples), ", Training Accuracy = ", _acc,
                      ", Minibatch Loss = ", _loss)
            step += 1
            g_step += 1
        valid_data = np.array(X_valid).reshape((-1, n_steps, n_input))
        vaild_label = y_valid
        file1.write("####################   test   #####################\n")
        test_epoch(valid_data, vaild_label, l_valid)
        epoch_num += 1
        X_train, y_train, l_train = epoch_shuffle(X_train, y_train, l_train)

        file1.write("####################   valid   #####################\n")
        test_data = np.array(X_test).reshape((-1, n_steps, n_input))
        f1_score, recall = test_epoch(test_data, y_test, l_test)

        if epoch_num % 10 == 0:
            saver.save(sess, "./model/testmodel.ckpt", write_meta_graph=False)
    print("Optimization Finished!")






