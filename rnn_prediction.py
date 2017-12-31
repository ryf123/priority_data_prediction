
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import random


# In[2]:


# time sequence, we look back 24 months
sequence_length = 24
# two result, advance or not
num_classes = 2
# define placeholder
inputs = tf.placeholder(tf.float64, [None, sequence_length, 1])
targets = tf.placeholder(tf.float64, [None, num_classes])


# In[3]:


# define cell
hidden_size = 32 # number of hidden state, when the cell outputs, the number of states
cell = tf.nn.rnn_cell.GRUCell(hidden_size)


# In[4]:


outputs, last_states = tf.nn.dynamic_rnn(
    cell,
    inputs,
    dtype=tf.float64)
outputs = tf.transpose(outputs, perm=[1, 0, 2])


# In[5]:


weights = tf.Variable(tf.random_normal([hidden_size, num_classes], dtype = tf.float64))
biases = tf.Variable(tf.random_normal([num_classes], dtype = tf.float64))
pred = tf.matmul(outputs[-1], weights) + biases


# In[6]:


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=targets))
# cost = tf.square(pred - targets)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(targets,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64))


# In[7]:


# data = [[[1.0],[2.0],[3.0]], [[1.0],[3.0],[2.0]]]
# data_output = [[1.0], [0.0]]
df = pd.read_csv("data/pd_data.csv")
advance = df['advance'].apply(lambda x: float(x == True)).tolist()
data = []
data_output = []
for i in range(sequence_length, len(advance)):
    a = []
    for s in advance[i-sequence_length:i]:
        a.append([s])
    data.append(a)
    if advance[i] == 0:
        data_output.append([1.0, 0.0])
    else:
        data_output.append([0.0, 1.0])


# In[8]:


# train/test split
# get random index
test_size = 12
test_index = random.sample(range(0, len(data)), test_size)
# sublist with the random index
train_data, train_output, test_data, test_output = [], [], [], []
for i, _ in enumerate(data):
    if i in test_index:
        test_data.append(data[i])
        test_output.append(data_output[i])
    else:
        train_data.append(data[i])
        train_output.append(data_output[i])


# In[15]:


with tf.Session() as sess:
    # run train data
    feed_dict = {inputs: train_data, targets: train_output}
    sess.run(tf.global_variables_initializer()) 
    for i in range(300):
        sess.run(optimizer, feed_dict=feed_dict)
        if i % 30 == 0:
            print("Training iteration:", i, " accuracy: ", sess.run(accuracy, feed_dict=feed_dict))
    # run test data
    feed_dict = {inputs: test_data, targets: test_output}
    print("Testing accuracy:", sess.run(accuracy, feed_dict=feed_dict))


# In[10]:


test_index

