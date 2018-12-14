""" Multilayer Perceptron.
A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function

# Import MNIST data
from project_utils import load_features, load_test_features
import tensorflow as tf
import numpy as np

preprocesses = ["None", "gaussian", "sqeeze"]
attacks = ["None", "spatial_grid", "fgsm", "gsma", "spatial_grid"]
   
x_train, y_train = load_features()

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 128
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = x_train.shape[1] # MNIST data input 
n_classes = y_train.shape[1] # MNIST total classes (6,7 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()


def train():
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(y_train)/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                # batch_x, batch_y = mnist.train.next_batch(batch_size)
                batch_x = x_train[i*batch_size:min((i+1)*batch_size,len(y_train))]
                batch_y = y_train[i*batch_size:min((i+1)*batch_size,len(y_train))]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                                Y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")
        for preprocess in preprocesses:
            for attack in attacks:
                print(preprocess, attack)
                feature, y_label = load_test_features(preprocess, attack)
                # Test model
                pred = tf.nn.softmax(logits)  # Apply softmax to logits
                # run
                prediction = pred.eval({X: feature, Y: y_label})     
                # calculate metric
                true_labels = np.argmax(y_label, axis=1)
                pred_labels = np.argmax(prediction[:len(y_label)], axis=1)
                # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
                TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
                
                # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
                TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
                
                # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
                FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
                
                # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
                FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

                recall = TP/float(TP+FN)
                precision = TP/float(TP+FP)
                f1 = 2.*TP/(2.*TP + FP + FN)  

                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print("Accuracy:", accuracy.eval({X: x_train, Y: y_train}))
                print(preprocess, attack, "accuracy", accuracy)
                print(preprocess, attack, "precision", precision)
                print(preprocess, attack, "recall", recall)
                print(preprocess, attack, "f1", f1)

        # Save the variables to disk.
        # save_path = saver.save(sess, "../models/mighty.ckpt")

def evaluate():
    preprocesses = ["None", "gaussian", "sqeeze"]
    attacks = ["None", "spatial_grid", "fgsm", "gsma", "spsa"]
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, tf.train.latest_checkpoint('../models/'))
        for preprocess in preprocesses:
            for attack in attacks:
                print(preprocess, attack)
                feature, y_label = load_test_features(preprocess, attack)
                logits = sess.run('Softmax', 
                     feed_dict={X: feature, Y: y_label})
                print(logits)
                
                # Test model
                pred = tf.nn.softmax(logits)  # Apply softmax to logits
                # run
                prediction = pred.eval({X: feature, Y: y_label})     
                # calculate metric
                true_labels = np.argmax(y_label, axis=1)
                pred_labels = np.argmax(prediction[:len(y_label)], axis=1)
                # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
                TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
                
                # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
                TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
                
                # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
                FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
                
                # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
                FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

                recall = TP/float(TP+FN)
                precision = TP/float(TP+FP)
                f1 = 2.*TP/(2.*TP + FP + FN)  
                print(preprocess, attack, "accuracy", accuracy)
                print(preprocess, attack, "precision", precision)
                print(preprocess, attack, "recall", recall)
                print(preprocess, attack, "f1", f1)

                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print("Accuracy:", accuracy.eval({X: feature, Y: y_label}))
train()
#evaluate()