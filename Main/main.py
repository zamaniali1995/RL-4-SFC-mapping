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
import numpy as np
input_cons = InputConstants.Inputs()
graph = Graph(input_cons.network_path + input_cons.network_name)
_chain = Chains(input_cons.chains_path + input_cons.chains_name, graph)
chains = _chain.read()

#_chain.function_cpu_usage(1)
#graph.function_placement(node=2, ser='WebService', fun='ali1')
#graph.function_placement(node=2, fun='ali')
#%%
#mf = Mf(graph)
graph.get_feature_matrix()
#mf.function_placement(node=2, ser='WebService', fun='ali')
#
#mf.function_placement(node=2, ser='WebService', fun='ali1')
#%%
#graph.floydWarshall()  
  

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
                                                input_cons.learning_rate)
opt = optimizer.minimize(cost)
init = tf.initialize_all_variables()
#%%
#grad = optimizer.compute_gradients(cross_entropy, w)
#gradients = [g for g, variable in grad]
#gradients1 = gradients[0]
#gradi_list = tf.zeros(input_cons.node_features)
#gradi_sum = tf.zeros(input_cons.node_features)
#        
#for i in range(input_cons.node_features):
#    tf.assign(gradi_list[i], gradients1[i][0])
#gradi_sum += gradi_list
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]
#gradient_placeholders = []
#grads_and_vars_feed = []
#for grad, variable in grads_and_vars:
#    gradient_placeholder = tf.placeholder(tf.float32, shape=grad[0].shape())
#    gradient_placeholders.append(gradient_placeholder)
#    grads_and_vars_feed.append((gradient_placeholder, variable))
#training_op = optimizer.apply_gradients(grads_and_vars_feed)

#training_op = optimizer.apply_gradients(grad)
# Session
trainAccList = []
testAccList = []
trainErrList = []
testErrList = []
#for s in range(len(chains)):
#    for f in range(len(chains[s].fun)):
#        fun = chains[s].fun[f]
#%%        
with tf.Session() as sess:
    init.run()
    for epoch in range(input_cons.epoch_num):
#        gradi_list = np.zeros(input_cons.node_features)
#        gradi_sum = np.zeros(input_cons.node_features)
        node_fun = []
        
        grads_stack = 0
        cnt = 0
        for ser_num, s  in enumerate(chains):
            ser_name = s.name
#            fun = chains[s].fun[f]        
            grad_stack = 0
            for fun in chains[ser_num].fun:
                graph.update_feature_matrix(node_fun)
            #    for epoch in range(1):
#                train_loss = 0
#                w_RL = sess.run(w)
#                logit_RL = sess.run(logit, feed_dict={x: mf.mf_matrix})
                y_RL = sess.run(y_predicted, feed_dict={x: graph.mf_matrix})
                y_one_hot, candidate = graph.select_one(y_RL,
                                                     approach='sample')
#                gradi = sess.run(grad, feed_dict={y: y_one_hot, x: mf.mf_matrix})
                                                       
#                print (gradi)
#                InputList = {x: mf.mf_matrix}
                gradients_val = sess.run(gradients, feed_dict={x: graph.mf_matrix, 
                                                  y: y_one_hot})
#                grad_stack += gradients_val
#                node_fun.append((candidate, fun))    
#                print(gradients_val)
#            if (graph.node_is_mapped(node_fun) & 
#                graph.link_is_mapped(node_fun)):
#                reward = 2
##                reward = rev_to_cost(node_fun)
#                grads_stack += (input_cons.learning_rate * reward 
#                               * grad_stack) 
#            else:
#                grads_stack = 0
#            cnt += 1
#            if cnt == input_cons.batch_Size:
#                #apply gradients
#                print(grads_stack)
#                cnt = 0
#                grads_stack = 0
#                
                
                
                
                
#                for i in range(input_cons.node_features):
#                    gradi_list[i] = gradi[i][0]
#                gradi_sum += gradi_list
#            print(gradi)
#            sess.run(optimizer.apply_gradients(gradi_sum))                                      

#                _, loss = sess.run([opt, cost], feed_dict=InputList)
#                print(loss)
#            node_fun.append((candidate, fun))
#        if (graph.node_check(node_fun) & graph.link_check(node_fun)):
            # apply gradients
#            pass
            #calculate reward
#            graph.batch_function_placement(ser=s,
#                                           node_fun=node_fun)
                
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