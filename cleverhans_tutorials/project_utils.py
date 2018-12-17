from cleverhans.dataset import MNIST_67
from six.moves import xrange
import numpy as np
import pickle
import logging
import tensorflow as tf
from  scipy import ndimage
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval, silence
from cleverhans.serial import load
from cleverhans.attacks import FastGradientMethod, SaliencyMapMethod, SPSA
import matplotlib.pyplot as plt

model_paths = ["../models/MNIST_FGSM_.joblib", "../models/MNIST_FGSM_gaussian.joblib", 
                "../models/MNIST_FGSM_sqeeze.joblib",
                "../models/MNIST_blackbox_.joblib", "../models/MNIST_blackbox_gaussian.joblib", 
                "../models/MNIST_blackbox_sqeeze.joblib"] 
attacks = [None, "spatial_grid", "fgsm", "jsma"]
preprocesses = [None, "gaussian", "sqeeze"] 
def test_data_sampling(x_path, y_path, size=2000):
    with open(x_path, "rb") as f:
        x_test = pickle.load(f)
    with open(y_path, "rb") as f:
        y_test_tf = pickle.load(f)
    y_test = np.zeros((len(y_test_tf),2))
    for i in range(len(y_test_tf)):
        if y_test_tf[i]:
            y_test[i, 1] = 1
        else:
            y_test[i, 0] = 1
    if(x_test.shape[ 0 ] > size):
        ind = np.arange( x_test.shape[ 0 ] )
        np.random.shuffle( ind )
        x_test = x_test[ ind[ :size ] ]
        y_test = y_test[ ind[ :size ] ]
    with open(x_path, 'wb') as handle:
      pickle.dump(x_test, handle)
    with open(y_path, 'wb') as handle:
      pickle.dump(y_test, handle)
  
def evaluate_model(filepath, attack=None, preprocess=None,
                    batch_size=128, num_threads=None):
  """
  Run evaluation on a saved model
  :param filepath: path to model to evaluate
  :param batch_size: size of evaluation batches
  """

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.INFO)

  # Create TF session
  if num_threads:
    config_args = dict(intra_op_parallelism_threads=1)
  else:
    config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))

  # Get MNIST test data
  x_train, y_train, x_test, y_test = get_MNIST_67_preprocess(
      test_attack=attack)

  # Use Image Parameters
  img_rows, img_cols, nchannels = x_test.shape[1:4]
  nb_classes = y_test.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  eval_params = {'batch_size': batch_size}

  def do_eval(preds, x_set, y_set, report_key, is_adv=None):
    acc = model_eval(sess, x, y, preds, x_set, y_set, save_logit=True, 
                    filename=report_key, args=eval_params)
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'legitimate'
    if report_text:
      print('Test accuracy on %s examples: %0.4f' % (report_text, acc))

  # Load Model
  with sess.as_default():
    model = load(filepath)
  assert len(model.get_params()) > 0

  # Attack
  if attack == 'fgsm':
    # Initialize the Fast Gradient Sign Method (FGSM) attack object and
    # graph
    # fgsm attack
    fgsm_params = {
        'eps': 0.3,
        'clip_min': 0.,
        'clip_max': 1.
    }
    fgsm = FastGradientMethod(model, sess=sess)
    # Loop over the samples we want to perturb into adversarial examples
    for sample_ind in xrange(0, len(x_test)):
        sample = x_test[sample_ind:(sample_ind + 1)]
        adv_x = fgsm.generate_np(sample, **fgsm_params)
        x_test[sample_ind:(sample_ind + 1)] = adv_x
        
  elif attack == 'jsma':
    jsma = SaliencyMapMethod(model, sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                 'clip_min': 0., 'clip_max': 1.,
                 'y_target': None}# Loop over the samples we want to perturb into adversarial examples
    for sample_ind in xrange(0, len(x_test)):
        sample = x_test[sample_ind:(sample_ind + 1)]
        one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
        if y_test[sample_ind, 0] == 1:
            one_hot_target[0, 1] = 1
        else:
            one_hot_target[0, 0] = 1
        jsma_params['y_target'] = one_hot_target
        adv_x = jsma.generate_np(sample, **jsma_params)
        x_test[sample_ind:(sample_ind + 1)] = adv_x
    plt.imshow(x_test[1], cmap='gray')
    plt.show()
    exit()
  # Image Process
  x_test = image_process(x_test, preprocess)

  preds = model.get_logits(x)
  fn = str(filepath[10:-7])+"_"+str(preprocess)+"_"+str(attack)
  do_eval(preds, x_test, y_test, fn, True)
  with open('{}_y.pickle'.format(fn), 'wb') as handle:
      pickle.dump(y_test, handle)

def load_test_features(preprocess, attack):
    feature = None
    y_label = None
    model_names = ["MNIST_FGSM_","MNIST_FGSM_gaussian","MNIST_FGSM_sqeeze", 
                "MNIST_blackbox_","MNIST_blackbox_gaussian","MNIST_blackbox_sqeeze" ]
    for model_name in model_names:
        filename = '../pickle/{}_{}_{}_.pickle'.format(
            model_name, preprocess, attack)
        with open(filename, "rb") as f:
            logits = pickle.load(f)
        if feature is None:
            feature = logits
        else:
            feature = np.hstack((feature, logits))
    if attack != 'spatial_grid':
        with open("../pickle/MNIST_FGSM_y_test.pickle", "rb") as f:
            y_label = pickle.load(f)
    else:
        with open("../pickle/attacked_spatial_grid_y.pickle", "rb") as f:
            y_label = pickle.load(f)
    return feature, y_label
            



def load_features():
    # fgsm features
    with open("../pickle/MNIST_FGSM_train_adv_train_adv_eval_.pickle", "rb") as f:
        fgsm_feature = pickle.load(f)
    with open("../pickle/MNIST_FGSM_train_adv_train_adv_eval_gaussian.pickle", "rb") as f:
        fgsm_feature = np.hstack((fgsm_feature, pickle.load(f)))
    with open("../pickle/MNIST_FGSM_train_adv_train_adv_eval_sqeeze.pickle", "rb") as f:
        fgsm_feature = np.hstack((fgsm_feature, pickle.load(f)))
    with open("../pickle/MNIST_FGSM_y_train.pickle", "rb") as f:
        fgsm_label = pickle.load(f)

    with open("../pickle/MNIST_blackbox_y_train.pickle", "rb") as f:
        blackbox_label = pickle.load(f)

    # black box features
    with open("../pickle/MNIST_blackbox_train_adv_eval_.pickle", "rb") as f:
        blackbox_feature = pickle.load(f)
    with open("../pickle/MNIST_blackbox_train_adv_eval_gaussian.pickle", "rb") as f:
        blackbox_feature = np.hstack((blackbox_feature, pickle.load(f)))
    with open("../pickle/MNIST_blackbox_train_adv_eval_sqeeze.pickle", "rb") as f:
        blackbox_feature = np.hstack((blackbox_feature, pickle.load(f)))

    size = min(len(blackbox_label), len(fgsm_label))
    feature = np.hstack((fgsm_feature[:size], blackbox_feature[:size]))
    return feature, blackbox_label

def image_process(x, preprocess=None):
    if preprocess == "gaussian":
        return apply_gaussian_filter_3d(x)
    elif preprocess == "sqeeze":
        return ndimage.filters.median_filter(x, size=(1,1,2,2), mode='reflect')
    return x

def get_MNIST_67_preprocess(test_attack=None, preprocess=None):
    mnist = MNIST_67(train_start=0, train_end=60000, test_start=0,
                   test_end=10000)
    x_train, y_train = mnist.get_set('train')
    x_test, y_test = mnist.get_set('test')

    if test_attack == 'spatial_grid':
        with open("../pickle/attacked_spatial_grid_x.pickle", "rb") as f:
            x_test = pickle.load(f)
        with open("../pickle/attacked_spatial_grid_y.pickle", "rb") as f:
            y_test = pickle.load(f)
            
    if test_attack is None:
        x_train = image_process(x_train, preprocess)
        x_test = image_process(x_test, preprocess)
    return x_train, y_train, x_test, y_test

def apply_gaussian_filter_3d(images, sigma = 2):
    # https://github.com/uvasrg/FeatureSqueezing
    for i in range(images.shape[0]):
        blur = ndimage.gaussian_filter(images[i], sigma=sigma)
        images[i] = blur
    return images

# test_data_sampling("../pickle/attacked_spatial_grid_x.pickle", "../pickle/attacked_spatial_grid_y.pickle")

def test_all_models():
    for model_path in model_paths:
        for p in preprocesses:
            for att in attacks:
                print(model_path, p, att)
                evaluate_model(model_path, attack=att, preprocess=p)

def baseline():
    for model_path in ["../models/CNN_gaussian.joblib", "../models/CNN_sqeeze.joblib"]:
        for p in preprocesses:
            for att in attacks:
                print(model_path, p, att)
                evaluate_model(model_path, attack=att, preprocess=p)
#test_all_models()
baseline()