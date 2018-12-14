# Import MNIST data
from project_utils import load_test_features
import tensorflow as tf

preprocesses = ["None", "gaussian", "sqeeze"]
attacks = ["None", "spatial_grid", "fgsm", "gsma", "spsa"]
# Add ops to save and restore all the variables.
saver = tf.train.import_meta_graph('../models/mighty.ckpt.meta')

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, tf.train.latest_checkpoint('../models/'))
    sess.run([train_op, loss_op], feed_dict={X: batch_x,Y: batch_y})
    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    # run
    pred.eval({X: x_train, Y: y_train})
    # calculate metric
    true_labels = np.argmax(Y_test, axis=1)
    pred_labels = np.argmax(prediction[:len(Y_test)], axis=1)
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
    print(filename, "accuracy", accuracy)
    print(filename, "precision", precision)
    print(filename, "recall", recall)
    print(filename, "f1", f1)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: x_train, Y: y_train}))
