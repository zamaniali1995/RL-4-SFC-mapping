#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 16:44:41 2019
@author: ali
"""
import sys
sys.path.insert(0, '../PaperFunctions')
#sys.path.insert(0, '../MFMatrix')
sys.path.insert(1, '../a')


import InputConstants
#from InputConstants import Inputs
#from PaperFunctions import Graph, Chains
#import tensorflow as tf
##from mfMatrix import Mf
#import numpy as np
##input_cons = InputConstants.Inputs()
#
##graph = Graph(input_cons.network_path + input_cons.network_name)
#_chain = Chains()
#_chain.creat_chains_functions(input_cons.chains_random_path + input_cons.chains_random_name,
#                     input_cons.chains_num,
#                     input_cons.fun_num,
#                     input_cons.chain_ban, 
#                     input_cons.cpu_ra         nge)
#functions = _chain.read_funcions(input_cons.chains_random_path + input_cons.chains_random_name)
#graph = Graph(input_cons.network_path + input_cons.network_name, 
#              functions)
##_chain.creat_chains(input_cons.chains_random_path + input_cons.chains_random_name,
##                     input_cons.chains_num,
##                     input_cons.fun_num,
##                     input_cons.chain_ban)
#chains = _chain.read_chains(input_cons.chains_random_path + input_cons.chains_random_name, 
#                     graph)
#graph.get_feature_matrix()
#%%
# Learning
tf.reset_default_graph()
node_num = len(graph.node_list)
# Def.ine placeholder x for input
x = tf.placeholder(dtype=tf.float64, shape=[node_num, input_cons.node_features], name="x")
# Define placeholder y for output
y = tf.placeholder(dtype=tf.float64, shape=[1, node_num], name="y")

# Define variable w and fill it with random number
w = tf.Variable(tf.random_normal(shape=[input_cons.node_features, 1], stddev=1e-15, mean=1e-15, dtype=tf.float64), name="weights", dtype=tf.float64, trainable=True)
#w = tf.get_variable(dtype=tf.float64, name="weights",
#                          initializer=tf.zeros(
#                                  shape=[input_cons.node_features, 1],
#                                  dtype=tf.float64))
## Define variable b and fill it with zero 
b = tf.Variable(tf.zeros(1, dtype=tf.float64), name="bias", dtype=tf.float64, trainable=True)
reward = tf.Variable(0, name="reward", dtype=tf.float64)
# Fetch a list of our network's trainable parameters.
trainable_vars = tf.trainable_variables()
# Create variables to store accumulated gradients
# Define logistic Regression
logit = tf.matmul(x, w) + b
logit_mod = tf.reshape(logit, [1, -1])
y_predicted = tf.nn.softmax(logit_mod)
# Define maximum likelihood loss function
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_mod, labels=y)
cost = tf.reduce_mean(loss)
# Define optimizer: GradientDescent         
optimizer = tf.train.GradientDescentOptimizer(learning_rate= input_cons.learning_rate)

# Compute gradients; grad_pairs contains (gradient, variable) pairs
grad_pairs = optimizer.compute_gradients(loss, [w, b])
opt = optimizer.minimize(cost)
init = tf.initialize_all_variables()
trainAccList = []
testAccList = []
trainErrList = []
testErrList = []

accumulators = [
    tf.Variable(
        tf.zeros_like(tv.initialized_value()),
        trainable=False
    ) for tv in [w, b]
    ]

accumulators_stacked = [
    tf.Variable(
        tf.zeros_like(tv.initialized_value()),
        trainable=False
    ) for tv in [w, b]
    ]
      
accumulate_ops = [
    accumulator.assign_add(
        grad 
    ) for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)
                    ]
accumulate_stacked_ops = [
    accumulator_s.assign_add(
        accumulator 
    ) for accumulator_s, accumulator in zip(accumulators_stacked,  accumulators)]
                                            
                           

accumulate_mul = [
    accumulator.assign_add(
        (accumulator * reward) - accumulator  
    ) for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)
                    ]
#accumulate_mul = [
#    tf.multiply(
#        accumulator, graph.rev_to_cost_val  
#    ) for accumulator in accumulators
#                ]
train_step = optimizer.apply_gradients(
    [(accumulator, var) 
        for (accumulator, (grad, var)) in zip(accumulators_stacked, grad_pairs)]
                        )

zero_ops = [
    accumulator.assign(
        tf.zeros_like(tv)
    ) for (accumulator, tv) in zip(accumulators, [w, b])]

zero_stacked_ops = [
    accumulator.assign(
        tf.zeros_like(tv)
    ) for (accumulator, tv) in zip(accumulators_stacked, [w, b])
                ]
with tf.Session() as sess:
    train_cnt = 0
    reward_list = []
    reward_list_final = []
    cost_list_final = []
    cost_list = []
    rev_list = []
    rev_list_final = []
    loss_list = []
    loss_list_final = []
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    for epoch in range(input_cons.epoch_num):
        placed_chains = []
        graph.make_empty_nodes()

        grads_stack = 0
        cnt = 0
        for ser_num, s  in enumerate(chains):
            node_fun = []
            ser_name = s.name

#            grad_stack = np.zeros([1, input_cons.node_features], dtype=np.float)
            for fun in s.fun:
                y_RL = sess.run(y_predicted, feed_dict={x: graph.mf_matrix})
                y_one_hot, candidate = graph.select_one(y_RL,
                                                     approach='sample')
                loss_list.append(sess.run(cost, feed_dict={y:y_RL , x: graph.mf_matrix}))
                gradient_val = sess.run(accumulate_ops, feed_dict={x: graph.mf_matrix, y: y_one_hot})
                node_fun.append((candidate, fun)) 
                graph.update_feature_matrix(node_fun)
                mf_matrix = graph.mf_matrix

            cnt += 1
            if (graph.node_is_mapped(node_fun, chains) & 
                graph.link_is_mapped(node_fun)):
#                reward = 0.001
                reward_val = graph.rev_to_cost(node_fun, ser_num, chains)
#                reward_val = graph.rev_to_cost_val
#                reward_tensor = tf.convert_to_tensor(reward_val)
                accu = sess.run(accumulators)
                sess.run(accumulate_mul, feed_dict={reward: reward_val})
                accu_mul = sess.run(accumulators)
                sess.run(accumulate_stacked_ops)
                accu_stack = sess.run(accumulators_stacked)                          
                reward_list.append(graph.rev_to_cost_val)
                cost_list.append(graph.cost_measure(node_fun,
                                                    ser_num, chains, 2))
                rev_list.append(graph.revenue_measure(node_fun, ser_num,
                                                      chains, 2))
                placed_chains.append(node_fun)
                sess.run(zero_ops) 
                accu_zero = sess.run(accumulators)
            else:
                loss_list = []
                placed_chains = []
                sess.run(zero_ops)
                sess.run(zero_stacked_ops)
                cnt = 0

            if cnt == input_cons.batch_Size:
                reward_list_final.append(sum(reward_list) / cnt)
                reward_list = []
                loss_list_final.append(sum(loss_list) / cnt)
                loss_list = []
                cost_list_final.append(sum(cost_list) / cnt)
                cost_list = []
                rev_list_final.append(sum(rev_list) / cnt)
                rev_list = []
                train_cnt += 1
                print("epoch = ", epoch)
                print("Train cnt = ", train_cnt)
                #apply gradients
#                accu = sess.run(accumulators)
                sess.run(train_step)
                w_val, b_val = sess.run([w, b])
#                print(accu_stack[1])
                print("b_val = ", b_val)
                print("w[0]_val = ", w_val[0])
                print("w[1]_val = ", w_val[1])
                print("w[2]_val = ", w_val[2])
                print("w[3]_val = ", w_val[3])
                print("**********************")
                graph.batch_function_placement(ser_name, placed_chains)
                placed_chains = []
                cnt = 0
                sess.run(zero_ops)
                accu_zero = sess.run(accumulators)
                sess.run(zero_stacked_ops)
                accu_stack_zero = sess.run(accumulators_stacked)                          
                
                