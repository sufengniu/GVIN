import numpy as np
import tensorflow as tf
from utils import *

class VI_Block():

  def __init__(self, scope, X, S1, S2, config):
    self.scope = scope
    self.X = X
    self.S1 = S1
    self.S2 = S2
    self.config = config

  def conv2d_flipkernel(x, k, name=None):
    return tf.nn.conv2d(x, flipkernel(k), name=name,
                        strides=(1, 1, 1, 1), padding='SAME')

  def build_model(self):
    with tf.variable_scope(self.scope):
      return self.build_VI_Block()      

  def build_VI_Block(self):
    k    = self.config.k    # Number of value iterations performed
    ch_i = self.config.ch_i # Channels in input layer
    ch_h = self.config.ch_h # Channels in initial hidden layer
    ch_q = self.config.ch_q # Channels in q layer (~actions)
    state_batch_size = self.config.statebatchsize # k+1 state inputs for each channel

    bias  = tf.Variable(np.random.randn(1, 1, 1, ch_h)    * 0.01, dtype=tf.float32)
    # weights from inputs to q layer (~reward in Bellman equation)
    w0    = tf.Variable(np.random.randn(3, 3, ch_i, ch_h) * 0.01, dtype=tf.float32)
    w1    = tf.Variable(np.random.randn(1, 1, ch_h, 1)    * 0.01, dtype=tf.float32)
    w     = tf.Variable(np.random.randn(3, 3, 1, ch_q)    * 0.01, dtype=tf.float32)
    # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
    w_fb  = tf.Variable(np.random.randn(3, 3, 1, ch_q)    * 0.01, dtype=tf.float32)
    w_o   = tf.Variable(np.random.randn(ch_q, 8)          * 0.01, dtype=tf.float32)

    # initial conv layer over image+reward prior
    h = conv2d_flipkernel(self.X, w0, name="h0") + bias

    r = conv2d_flipkernel(h, w1, name="r")
    q = conv2d_flipkernel(r, w, name="q")
    v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    for i in range(0, k-1):
      rv = tf.concat([r, v], 3)
      wwfb = tf.concat([w, w_fb], 2)
      q = conv2d_flipkernel(rv, wwfb, name="q")
      self.v_ = tf.reduce_max(q, axis=3, keep_dims=True, name="v")
      v = self.v_

    
    # do one last convolution
    q = conv2d_flipkernel(tf.concat([r, v], 3),
                          tf.concat([w, w_fb], 2), name="q")

    # CHANGE TO THEANO ORDERING
    # Since we are selecting over channels, it becomes easier to work with
    # the tensor when it is in NCHW format vs NHWC
    q = tf.transpose(q, perm=[0, 3, 1, 2])
    v = tf.transpose(v, perm=[0, 3, 1, 2])
    # Select the conv-net channels at the state position (S1,S2).
    # This intuitively corresponds to each channel representing an action, and the convnet the Q function.
    # The tricky thing is we want to select the same (S1,S2) position *for each* channel and for each sample
    # TODO: performance can be improved here by substituting expensive
    #       transpose calls with better indexing for gather_nd
    bs = tf.shape(q)[0]
    rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
    ins1 = tf.cast(tf.reshape(self.S1, [-1]), tf.int32)
    ins2 = tf.cast(tf.reshape(self.S2, [-1]), tf.int32)
    idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])

    q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in, name="q_out")
    v_out = tf.transpose(self.extract_circle(rprn, ins1, ins2, v), [1,0,2])
    v_out = tf.squeeze(v_out)
    # add logits
    logits1 = tf.matmul(q_out, w_o)
    logits2 = v_out
    # softmax output weights
    output1 = tf.nn.softmax(logits1, name="output")
    output2 = tf.nn.softmax(logits2, name="output")
    
    return logits1, output1, v, logits2, output2


  def extract_circle(self, rprn, ins1, ins2, v):
    circle = []

    ins1_ = ins1 - 1  
    ins2_ = ins2 
    idx_in = tf.transpose(tf.stack([ins1_, ins2_, rprn]), [1, 0])
    circle.append(tf.gather_nd(tf.transpose(v, [2, 3, 0, 1]), idx_in))
    ins1_ = ins1 + 1 
    ins2_ = ins2 
    idx_in = tf.transpose(tf.stack([ins1_, ins2_, rprn]), [1, 0])
    circle.append(tf.gather_nd(tf.transpose(v, [2, 3, 0, 1]), idx_in))
    ins1_ = ins1  
    ins2_ = ins2 + 1 
    idx_in = tf.transpose(tf.stack([ins1_, ins2_, rprn]), [1, 0])
    circle.append(tf.gather_nd(tf.transpose(v, [2, 3, 0, 1]), idx_in))
    ins1_ = ins1 
    ins2_ = ins2 - 1  
    idx_in = tf.transpose(tf.stack([ins1_, ins2_, rprn]), [1, 0])
    circle.append(tf.gather_nd(tf.transpose(v, [2, 3, 0, 1]), idx_in))
    # ins1_ = ins1
    # ins2_ = ins2
    # idx_in = tf.transpose(tf.stack([ins1_, ins2_, rprn]), [1, 0])
    # circle.append(tf.gather_nd(tf.transpose(v, [2, 3, 0, 1]), idx_in))
    ins1_ = ins1 - 1
    ins2_ = ins2 + 1
    idx_in = tf.transpose(tf.stack([ins1_, ins2_, rprn]), [1, 0])
    circle.append(tf.gather_nd(tf.transpose(v, [2, 3, 0, 1]), idx_in))
    ins1_ = ins1 - 1 
    ins2_ = ins2 - 1
    idx_in = tf.transpose(tf.stack([ins1_, ins2_, rprn]), [1, 0])
    circle.append(tf.gather_nd(tf.transpose(v, [2, 3, 0, 1]), idx_in))
    ins1_ = ins1 + 1 
    ins2_ = ins2 + 1
    idx_in = tf.transpose(tf.stack([ins1_, ins2_, rprn]), [1, 0])
    circle.append(tf.gather_nd(tf.transpose(v, [2, 3, 0, 1]), idx_in))
    ins1_ = ins1 + 1 
    ins2_ = ins2 - 1
    idx_in = tf.transpose(tf.stack([ins1_, ins2_, rprn]), [1, 0])
    circle.append(tf.gather_nd(tf.transpose(v, [2, 3, 0, 1]), idx_in))
    circle = tf.stack(circle)
    return circle
