import re
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

class Model():
    def __init__(self, word_vec, edge_word_vec, word_dim, edge_word_dim, max_length, class_num, learning_rate):
        self.word_vec = word_vec
        self.edge_word_vec = edge_word_vec
        self.word_dim = word_dim
        self.edge_word_dim = edge_word_dim
        self.max_length = max_length
        self.class_num = class_num
        self.gat_layer = 3
        self.conv_size = 1024
        self.learning_rate = learning_rate

        self.sentence_ids = tf.placeholder(tf.int32, [None, self.max_length], name='sentence_ids')
        self.label_ids = tf.placeholder(tf.int32, [None, self.max_length, self.max_length], name='label_ids')
        self.entity_index = tf.placeholder(tf.int32, [None, self.max_length], name='entity_index')
        self.y = tf.placeholder(tf.int32, [None, ], name='y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.label_masks = tf.placeholder(tf.float32, [None, self.max_length, self.max_length], name='label_masks')

        self.word_ebd = tf.get_variable(initializer=self.word_vec, name='word_embedding', trainable=False)
        self.edge_ebd = tf.get_variable(initializer=self.edge_word_vec, name='edge_embedding', trainable=False)

        self.hidden = self.gat()

        self.entity_ebd = tf.nn.embedding_lookup(self.word_ebd, self.entity_index)
        self.entity_ebd = tf.reduce_mean(self.entity_ebd, axis=1, keepdims=False)
        self.entity_ebd = tf.reshape(self.entity_ebd, [1,1,self.word_dim])

        self.inputs = tf.concat(axis=1, values=[self.hidden, self.entity_ebd])
        self.outputs = self.task_network(inputs=self.inputs)

        self.pre = tf.argmax(tf.nn.softmax(self.outputs), axis=1)

        self.neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.outputs,
                                                                           labels=self.y)
        self.loss = tf.reduce_mean(self.neg_log_prob)

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        return

    def gat(self):
        sentence_ebd = tf.nn.embedding_lookup(self.word_ebd, self.sentence_ids)
        label_ebd = tf.nn.embedding_lookup(self.edge_ebd, self.label_ids)

        h = []
        probs = []
        for lay_num in range(self.gat_layer):
            h_, prob_ = self.gpr(label_ebd=label_ebd, sentence_ebd=sentence_ebd, lay_num=lay_num)
            h.append(h_)
            probs.append(prob_)

        hidden = tf.reduce_mean(h, axis=0, keepdims=False)
        self.probs = probs

        return hidden

    def gpr(self, label_ebd, sentence_ebd, lay_num):
        self.conv_input = tf.reshape(label_ebd, [-1, self.max_length*self.max_length, self.edge_word_dim, 1])
        self.conv = layers.conv2d(inputs=self.conv_input, num_outputs=self.conv_size, kernel_size=[3, 50]
                                  , stride=[1, 50], padding='SAME')

        # self.conv_2 = tf.layers.conv2d(inputs=self.conv_input_1, filters=self.conv_size, kernel_size=[3,50], strides=[1,50], padding='SAME', use_bias=False)
        self.max_pool = layers.max_pool2d(self.conv, kernel_size=[self.max_length*self.max_length, 1], stride=[1, 1])
        self.conv_output = tf.reshape(self.max_pool, [-1, self.conv_size])

        self.conv_output = tf.nn.tanh(self.conv_output)
        self.conv_output = tf.nn.dropout(self.conv_output, keep_prob=self.keep_prob)

        w_name = 'gw' + str(lay_num)
        b_name = 'gb' + str(lay_num)

        W = self.weight_variable(name=w_name, shape=[self.conv_size, self.max_length*self.max_length])
        b = self.bias_variable(name=b_name, shape=[self.max_length*self.max_length])
        prob = tf.matmul(self.conv_output, W) + b
        prob = tf.reshape(prob, [-1, self.max_length, self.max_length])
        prob = tf.nn.softmax(prob, axis=1)
        prob = tf.multiply(prob, self.label_masks)

        h = tf.matmul(prob, sentence_ebd)

        return h, prob

    def task_network(self, inputs):
        self.conv_input = tf.reshape(inputs, [-1, self.max_length+1, self.word_dim, 1])
        self.conv = layers.conv2d(inputs=self.conv_input, num_outputs=self.conv_size, kernel_size=[3, 100],
                                  stride=[1, 100], padding='SAME')

        self.max_pool = layers.max_pool2d(self.conv, kernel_size=[self.max_length+1, 1], stride=[1, 1])
        self.conv_output = tf.reshape(self.max_pool, [-1, self.conv_size])

        self.conv_output = tf.nn.tanh(self.conv_output)
        self.conv_output = tf.nn.dropout(self.conv_output, keep_prob=self.keep_prob)

        W1 = self.weight_variable(name='tw1', shape=[self.conv_size, self.conv_size])
        b1 = self.bias_variable(name='t1', shape=[self.conv_size])
        h1 = tf.matmul(self.conv_output, W1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.nn.dropout(h1, keep_prob=self.keep_prob)

        W2 = self.weight_variable(name='tw2', shape=[self.conv_size, self.class_num])
        b2 = self.bias_variable(name='tb2', shape=[self.class_num])
        h2 = tf.matmul(h1, W2) + b2

        return h2

    def weight_variable(self, name, shape):
        initial = tf.random_normal_initializer(0., 0.3)
        return tf.get_variable(name=name, shape=shape, initializer=initial)

    def bias_variable(self, name, shape):
        initial = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initial)