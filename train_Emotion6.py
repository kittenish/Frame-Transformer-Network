import tensorflow as tf
import read_mat as read_mat
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot, weight_variable_cnn
from shuffle_data import shuffle, shuffle_emotion6
from transform import transformer

x_train = read_mat.read_mat_v('/home/g_jiarui/video_spacial/data/multi_train_data.mat', 'train_data')/200
y_train = read_mat.read_mat('/home/g_jiarui/video_spacial/data/multi_train_label.mat')['train_label']

X_valid = read_mat.read_mat('/home/g_jiarui/video_spacial/data/multi_validation_data.mat')['validation_data']/200
y_valid = read_mat.read_mat('/home/g_jiarui/video_spacial/data/multi_validation_label.mat')['validation_label'][0]

X_test = read_mat.read_mat('/home/g_jiarui/video_spacial/data/multi_test_data.mat')['test_data']/200
y_test = read_mat.read_mat('/home/g_jiarui/video_spacial/data/multi_test_label.mat')['test_label'][0]

position_train = read_mat.read_mat('/home/g_jiarui/video_spacial/data/multi_train_position.mat')['train_position']
position_val = read_mat.read_mat('/home/g_jiarui/video_spacial/data/multi_validation_position.mat')['validation_position']
np.savetxt("/home/g_jiarui/video_spacial/result/1/train/val_position.txt",position_val,fmt="%d")

x_train_trans = np.transpose(x_train)
[X_train,y_train,position] = shuffle_emotion6(x_train_trans, y_train, position_train, 2400)
Y_train = dense_to_one_hot(y_train, n_classes=6)
Y_valid = dense_to_one_hot(y_valid, n_classes=6)
Y_test = dense_to_one_hot(y_test, n_classes=6)

X_train = np.resize(X_train, (2400, 30, 4096, 1))
X_valid = np.resize(X_valid, (600, 30, 4096, 1))
X_test = np.resize(X_test, (600, 30, 4096, 1))

x = tf.placeholder(tf.float32, [None, 30, 4096, 1])
y = tf.placeholder(tf.float32,[None, 6])
keep_prob = tf.placeholder(tf.float32)
x_flat = tf.reshape(x, [-1, 30 * 4096])
x_trans = tf.reshape(x, [-1, 30, 4096, 1])

W_fc_loc1 = weight_variable([30 * 4096, 20])
b_fc_loc1 = bias_variable([20])

initial = np.array([0.5, 0])
initial = initial.astype('float32')
initial = initial.flatten()

W_fc_loc2 = weight_variable([20, 2])
b_fc_loc2 = bias_variable([2])
b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

h_fc_loc1 = tf.nn.tanh(tf.matmul(x_flat, W_fc_loc1) + b_fc_loc1)
h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

out_size = (10, 4096)
h_trans = transformer(x_trans, h_fc_loc2, out_size)
h_flat = tf.reshape(h_trans, [-1, 10, 4096, 1])

# start cnn

#filter_size = 3
filter_size = 5
n_filters_1 = 8
W_conv1 = weight_variable_cnn([filter_size, 1, 1, n_filters_1])
b_conv1 = bias_variable([n_filters_1])
h_conv1 = tf.nn.relu(
	tf.nn.conv2d(input=h_trans,
				filter=W_conv1,
				strides=[1, 1, 1, 1],
				padding='SAME') +
	b_conv1)

h_conv1_flat = tf.reshape(h_conv1, [-1, 10*4096*n_filters_1])

n_fc = 32
W_fc1 = weight_variable_cnn([10 * 4096 * n_filters_1, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.tanh(tf.matmul(h_conv1_flat/200, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable_cnn([n_fc, 6])
b_fc2 = bias_variable([6])
y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


'''
#1*3
#filter_size = 3
filter_size = 3
n_filters_1 = 8
W_conv1 = weight_variable_cnn([1, filter_size, 1, n_filters_1])
b_conv1 = bias_variable([n_filters_1])
h_conv1 = tf.nn.relu(
	tf.nn.conv2d(input=h_trans,
				filter=W_conv1,
				strides=[1, 1, 1, 1],
				padding='SAME') +
	b_conv1)

h_conv1_flat = tf.reshape(h_conv1, [-1, 10*4096*n_filters_1])

n_fc = 32
W_fc1 = weight_variable_cnn([10 * 4096 * n_filters_1, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.tanh(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable_cnn([n_fc, 6])
b_fc2 = bias_variable([6])
y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
'''

'''
#3*3
filter_size = 3
n_filters_1 = 16
W_conv1 = weight_variable_cnn([filter_size, filter_size, 1, n_filters_1])
b_conv1 = bias_variable([n_filters_1])

h_conv1 = tf.nn.relu(
	tf.nn.conv2d(input=h_trans,
				filter=W_conv1,
				strides=[1, 2, 2, 1],
				padding='SAME') +
	b_conv1)

h_conv1_flat = tf.reshape(h_conv1, [-1, 3 * 2048 * n_filters_1])
n_fc = 32
W_fc1 = weight_variable_cnn([3 * 2048 * n_filters_1, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.tanh(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable_cnn([n_fc, 2])
b_fc2 = bias_variable([2])
y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
'''


'''
#3*4096
filter_size = 3
n_filters_1 = 256
W_conv1 = weight_variable_cnn([filter_size, 4096, 1, n_filters_1])
b_conv1 = bias_variable([n_filters_1])

h_conv1 = tf.nn.relu(
    tf.nn.conv2d(input=h_trans,
                 filter=W_conv1,
                 strides=[1, 1, 1, 1],
                 padding='VALID') +
    b_conv1)

h_conv1_flat = tf.reshape(h_conv1, [-1, 4 * 1 * n_filters_1])

n_fc = 64
W_fc1 = weight_variable_cnn([4 * 1 * n_filters_1, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.tanh(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable_cnn([n_fc, 2])
b_fc2 = bias_variable([2])
y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
'''

'''
#fc_2
n_fc = 64
#W_fc1 = weight_variable_cnn([4 * 1 * n_filters_1, n_fc])
W_fc1 = weight_variable_cnn([30*4096, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.tanh(tf.matmul(x_flat, W_fc1) + b_fc1)
#h_fc1 = tf.nn.tanh(tf.matmul(h_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable_cnn([n_fc, 6])
b_fc2 = bias_variable([6])
y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
'''

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
opt = tf.train.AdamOptimizer()
optimizer = opt.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
grads = opt.compute_gradients(cross_entropy, [b_fc_loc2])
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.initialize_all_variables())

iter_per_epoch = 60
n_epochs = 1000
train_size = 2400

indices = np.linspace(0, train_size - 1, iter_per_epoch)
indices = indices.astype('int')
max_acc = 0;
out_theta = []
out_position = []
ff = open('/home/g_jiarui/video_spacial/result/1/train/accuracy.txt','w+')

for epoch_i in range(n_epochs):
    for iter_i in range(iter_per_epoch - 1):
        batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
        batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]
        
        if iter_i % 10 == 0:
            loss = sess.run(cross_entropy,
                            feed_dict={
                                x: batch_xs,
                                y: batch_ys,
                                keep_prob: 1.0
                            })
            print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))

        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.75})

    acc = str(sess.run(accuracy,feed_dict={
                                                         x: X_valid,
                                                         y: Y_valid,
                                                         keep_prob: 1.0
                                                     }))
    print('Accuracy (%d): ' %epoch_i + acc)
    print('test (%d): ' % epoch_i + str(sess.run(accuracy,
                                                     feed_dict={
                                                         x: X_test,
                                                         y: Y_test,
                                                         keep_prob: 1.0
                                                     })))
    
    #theta = sess.run(h_fc_loc2, feed_dict={
    #        x: batch_xs, keep_prob: 1.0})
    #print theta

    #grad_vals = sess.run([g for (g,v) in grads], feed_dict={
    #        x: batch_xs, y: batch_ys, keep_prob: 1.0})
    #print 'grad_vals: ', grad_vals

    if epoch_i % 5 == 1:
        for iter_i in range(iter_per_epoch - 1):
            batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
            batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]

            theta = sess.run(h_fc_loc2, feed_dict={
                x: batch_xs, keep_prob: 1.0})

            if iter_i == 0:
                out_theta = theta
            else:
                out_theta = np.concatenate((out_theta,theta),axis=0)
        
        np.savetxt("/home/g_jiarui/video_spacial/result/1/train_theta_result_"+str(epoch_i)+".txt",out_theta,fmt="%f")

        val_theta = sess.run(h_fc_loc2, feed_dict={
                x: X_valid, y: Y_valid, keep_prob: 1.0})

        np.savetxt("/home/g_jiarui/video_spacial/result/1/val_theta_result_"+str(epoch_i)+".txt",val_theta,fmt="%f")

        #saver.save(sess, '/home/hs/video_spacial/syhthetic/color/resul_10t/3*3_initial=0.5/synthetic_data.tfmodel');
        
        ff.write('Accuracy_' + str(epoch_i) + '_'  + str(acc) + '\n')

ff.close()

