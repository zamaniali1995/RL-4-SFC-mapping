#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 16:44:41 2019

@author: ali
"""
import sys
sys.path.insert(0, '../ReadFile')
sys.path.insert(0, '../MFMatrix')
sys.path.insert(0, '../Given')
import InputConstants
from ReadFile import Graph, Chains
import tensorflow as tf
from mfMatrix import Mf
#import numpy as np
input_cons = InputConstants.Inputs()
graph = Graph(input_cons.network_path + input_cons.network_name)
_chain = Chains(input_cons.chains_path + input_cons.chains_name, graph)
chains = _chain.read()
graph.function_placement(node=2, ser='WebService', fun='ali1')
#graph.function_placement(node=2, fun='ali')
#%%
mf = Mf(graph)
#mf.function_placement(node=2, ser='WebService', fun='ali')
#
#mf.function_placement(node=2, ser='WebService', fun='ali1')
#%%

# Learning

node_num = len(graph.node_list)
# Def.ine placeholder x for input
x = tf.placeholder(dtype=tf.float64, shape=[node_num, input_cons.node_features], name="x")
# Define placeholder y for output
y = tf.placeholder(dtype=tf.float64, shape=[1, node_num], name="y")
# Define variable w and fill it with random number
w = tf.Variable(tf.random_normal(shape=[input_cons.node_features, 1], stddev=0.1, dtype=tf.float64), name="weights", dtype=tf.float64)
# Define variable b and fill it with zero 
b = tf.Variable(tf.zeros(1, dtype=tf.float64), name="bias", dtype=tf.float64)
# Define logistic Regression
logit = tf.matmul(x, w) + b
logit_mod = tf.reshape(logit, [1, -1])
#_sum = tf.reduce_sum(logit)
#y_predicted = tf.exp(logit[0])/(1+_sum)
y_predicted = tf.nn.softmax(logit_mod)
#y_predicted1 = 1.0 / (1.0 + tf.exp(-logit))
# Define maximum likelihood loss function
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_mod, labels=y)
cost = tf.reduce_mean(cross_entropy)
# Define optimizer: GradientDescent         
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 
                                                input_cons.learning_rate).minimize(cost)
init = tf.initialize_all_variables()

# Session
trainAccList = []
testAccList = []
trainErrList = []
testErrList = []
for s in range(len(chains)):
    for f in range(len(chains[s].fun)):
        fun = chains[s].fun[f]
        
with tf.Session() as sess:
    init.run()
    for s in range(len(chains)):
        node_fun = []
        for fun in chains[s].fun:
#            fun = chains[s].fun[f]        
            mf_matrix = mf.get_feature_matrix()
            for epoch in range(input_cons.epoch_num):
            #    for epoch in range(1):
#                train_loss = 0
#                w_RL = sess.run(w)
#                logit_RL = sess.run(logit, feed_dict={x: mf.mf_matrix})
                y_RL = sess.run(y_predicted, feed_dict={x: mf.mf_matrix})
                y_one_hot, candidate = mf.select_one(y_RL,
                                                     approach='roulette_wheel')
#                InputList = {x: mf.mf_matrix,
#                     y: y_one_hot}
#                _, loss = sess.run([optimizer, cost], feed_dict=InputList)
#                print(loss)
            node_fun.append((candidate, fun))
        if graph.check(node_fun):
            graph.batch_function_placement(ser=s,
                                           node_fun=node_fun)
                
#        y_RL1 = sess.run(y_predicted1, feed_dict={x: mf.mf_matrix})
#       
#        y_one_hot = y_one_hot.reshape(1, -1)
        
#    for epoch in range(input_cons.epoch_num):
        
#    print(y_RL)
#        InputList = {x: m_f,
#                     y: }
#        _, loss = sess.run([optimizer, cost], feed_dict=InputList)
#        trainLoss += loss
#        trainPredicted = sess.run(yHat, feed_dict={x: dataTrain})
#        trainPredicted = np.argmax(trainPredicted, 1) + 1
#        trainErrList.append(trainLoss)
#        trainAccList.append(acc(targetTrain, trainPredicted))
#        print("lass:", epoch, trainLoss)
#        print("Train Acc", trainAccList[epoch])
#        testPredicted = sess.run(yHat, feed_dict={x: dataTest})
#        testPredicted = np.argmax(testPredicted, 1) + 1
#        testErrList.append(sess.run(cost, feed_dict={x: dataTest, yTrue: targetTestModifid}))
#        testAccList.append(acc(targetTest, testPredicted))
#        print("Test Acc", testAccList[epoch])
#    w1, b1, w2, b2, w3, b3, wO, bO = sess.run([wLayer1, bLayer1, wLayer2,
#                                               bLayer2, wLayer3, bLayer3, wLayerOut, bLayerOut])
#        
##print(graph.start_file_line)
#
##print (IndentationError.read_path)
##Mf_cal()
#            