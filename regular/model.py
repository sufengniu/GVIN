import numpy as np
import tensorflow as tf
from utils import conv2d_flipkernel, adjecent_matrix, adjecent_sparse


def dot(x, y, sparse=False):
    if sparse:
        return tf.sparse_tensor_dense_matmul(x, y)
    else:
        return tf.matmul(x, y)

def ir_Block(X, S1, S2, config):
    ka   = config.ka
    k    = config.k    # Number of value iterations performed
    t    = config.t
    ch_i = config.ch_i # Channels in input layer
    ch_h = config.ch_h # Channels in initial hidden layer
    ch_q = config.ch_q # Channels in q layer (~actions)
    state_batch_size = config.statebatchsize # k+1 state inputs for each channel
    img_s = config.imsize
    P = []
    P_fb = []

    # reference direction
    theta_init = np.array([np.pi*3.0/4.0, np.pi/2.0, np.pi/4.0, np.pi, 0.0, np.pi*5.0/4.0, np.pi*3.0/2.0, np.pi*7.0/4.0], dtype=np.float32)

    wi = [tf.Variable(np.random.randn(ka+1) * 0.01, dtype=tf.float32) for i in range(ch_q)]
    if config.fixed is True:
        thetai = [theta_init for i in range(ch_q)]
        theta_fb = [theta_init for i in range(ch_q)]
    else:
        thetai = [tf.Variable(np.random.random(ka) * 2*np.pi, dtype=tf.float32) for i in range(ch_q)]
        theta_fb = [tf.Variable(np.random.random(ka) * 2*np.pi, dtype=tf.float32) for i in range(ch_q)]

    # coefficients in paper eq. 5 (section 6.3) on forward path
    coeff = []
    for i in range(ch_q):
        coeff_tmp = []
        theta = 0.0
        for j in range(ka):
            coeff_tmp.append(tf.reduce_sum(wi[i][:ka]*tf.cast(tf.pow(tf.div(1.0+tf.cos(theta - thetai[i]), 2.0), t), dtype=tf.float32)) + wi[i][ka])
            theta += np.pi/4.0
        coeff_tmp.insert(4, wi[i][ka])
        coeff.append(tf.stack(coeff_tmp))

    # coefficients in paper eq. 5 (section 6.3) on feedback path
    w_fb = [tf.Variable(np.random.randn(ka+1) * 0.01, dtype=tf.float32) for i in range(ch_q)]
    coeff_fb = []
    for i in range(ch_q):
        coeff_tmp = []
        theta = 0.0
        for j in range(ka):
            coeff_tmp.append(tf.reduce_sum(w_fb[i][:ka]*tf.cast(tf.pow(tf.div(1.0+tf.cos(theta - thetai[i]), 2.0), t), dtype=tf.float32)) + w_fb[i][ka])
            theta += np.pi/4.0
        coeff_tmp.insert(4, w_fb[i][ka])
        coeff_fb.append(tf.stack(coeff_tmp))
    adj_M = adjecent_sparse(config.imsize, config.imsize)

    # obtain P (transition) and P_fb (transition for feedback channel)
    for j in range(ch_q):
        tmp_p = tf.sparse_add(tf.cast(tf.SparseTensor(adj_M[0][0], adj_M[0][1]*coeff[j][0], [img_s*img_s, img_s*img_s]), tf.float32),
            tf.cast(tf.SparseTensor(adj_M[1][0], adj_M[1][1]*coeff[j][1], [img_s*img_s, img_s*img_s]), tf.float32))
        tmp_p_fb = tf.sparse_add(tf.cast(tf.SparseTensor(adj_M[0][0], adj_M[0][1]*coeff_fb[j][0], [img_s*img_s, img_s*img_s]), tf.float32),
            tf.cast(tf.SparseTensor(adj_M[1][0], adj_M[1][1]*coeff_fb[j][1], [img_s*img_s, img_s*img_s]), tf.float32))
        for i in range(2, len(adj_M)):
            tmp_p = tf.sparse_add(tmp_p, tf.cast(tf.SparseTensor(adj_M[i][0], adj_M[i][1]*coeff[j][i], [img_s*img_s, img_s*img_s]), tf.float32))
            tmp_p_fb = tf.sparse_add(tmp_p_fb, tf.cast(tf.SparseTensor(adj_M[i][0], adj_M[i][1]*coeff_fb[j][i], [img_s*img_s, img_s*img_s]), tf.float32))
        P.append(tmp_p)
        P_fb.append(tmp_p_fb)

    bias  = tf.Variable(np.random.randn(1, 1, 1, ch_h)    * 0.01, dtype=tf.float32)
    # weights from inputs to q layer (~reward in Bellman equation)
    w0    = tf.Variable(np.random.randn(3, 3, ch_i, ch_h) * 0.01, dtype=tf.float32)
    w1    = tf.Variable(np.random.randn(1, 1, ch_h, 1)    * 0.01, dtype=tf.float32)

    # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
    # only used when config.v is False
    w_o   = tf.Variable(np.random.randn(ch_q, 8)          * 0.01, dtype=tf.float32)

    # initial conv layer over image+reward prior
    h = conv2d_flipkernel(X, w0, name="h0") + bias

    r = conv2d_flipkernel(h, w1, name="r")
    r = tf.reshape(r, [-1, img_s * img_s, 1])
    r_ = tf.reshape(r, [-1, img_s * img_s])

    q = []
    for i in range(ch_q):
        tmp = tf.transpose(tf.sparse_tensor_dense_matmul(P[i], tf.transpose(r_)))
        q.append(tmp)
    q = tf.transpose(tf.stack(q), [1,2,0])
    v = tf.reduce_max(q, axis=2, keep_dims=True, name="v")
    v_ = tf.reshape(v, [-1, img_s * img_s])

    for i in range(0, k-1):
        q1, q2 = [], []
        for i in range(ch_q):
            q1.append(tf.transpose(tf.sparse_tensor_dense_matmul(P[i], tf.transpose(r_))))
            q2.append(tf.transpose(tf.sparse_tensor_dense_matmul(P_fb[i], tf.transpose(v_))))
        q1 = tf.transpose(tf.stack(q1), [1,2,0])
        q2 = tf.transpose(tf.stack(q2), [1,2,0])
        q = q1+q2
        v = tf.reduce_max(q, axis=2, keep_dims=True, name="v")
        v_ = tf.reshape(v, [-1, img_s * img_s])

    # do one last convolution
    q1, q2 = [], []
    for i in range(ch_q):
        q1.append(tf.transpose(tf.sparse_tensor_dense_matmul(P[i], tf.transpose(r_))))
        q2.append(tf.transpose(tf.sparse_tensor_dense_matmul(P_fb[i], tf.transpose(v_))))
    q1 = tf.transpose(tf.stack(q1), [1,2,0])
    q2 = tf.transpose(tf.stack(q2), [1,2,0])
    q = q1+q2
    q = tf.reshape(q, [-1, img_s, img_s, ch_q])
    # CHANGE TO THEANO ORDERING
    # Since we are selecting over channels, it becomes easier to work with
    # the tensor when it is in NCHW format vs NHWC
    q = tf.transpose(q, perm=[0, 3, 1, 2])
    if config.v is True:
        v = tf.reshape(v, [-1, img_s, img_s, 1])
        v = tf.transpose(v, perm=[0, 3, 1, 2])

    # Select the conv-net channels at the state position (S1,S2).
    # This intuitively corresponds to each channel representing an action, and the convnet the Q function.
    # The tricky thing is we want to select the same (S1,S2) position *for each* channel and for each sample
    # TODO: performance can be improved here by substituting expensive
    #       transpose calls with better indexing for gather_nd
    bs = tf.shape(q)[0]
    rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
    ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
    ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
    idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])

    if config.v is True:
        v_out = tf.transpose(extract_circle(rprn, ins1, ins2, v), [1,0,2])
        v_out = tf.squeeze(v_out)
        logits = v_out
    else:
        q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in, name="q_out")
        logits = tf.matmul(q_out, w_o)
    
    # softmax output weights
    output = tf.nn.softmax(logits, name="output")
    return logits, output


def extract_circle(rprn, ins1, ins2, v):
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

def VI_Block(X, S1, S2, config):
    k    = config.k    # Number of value iterations performed
    ch_i = config.ch_i # Channels in input layer
    ch_h = config.ch_h # Channels in initial hidden layer
    ch_q = config.ch_q # Channels in q layer (~actions)
    state_batch_size = config.statebatchsize # k+1 state inputs for each channel

    bias  = tf.Variable(np.random.randn(1, 1, 1, ch_h)    * 0.01, dtype=tf.float32)
    # weights from inputs to q layer (~reward in Bellman equation)
    w0    = tf.Variable(np.random.randn(3, 3, ch_i, ch_h) * 0.01, dtype=tf.float32)
    w1    = tf.Variable(np.random.randn(1, 1, ch_h, 1)    * 0.01, dtype=tf.float32)
    w     = tf.Variable(np.random.randn(3, 3, 1, ch_q)    * 0.01, dtype=tf.float32)
    # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
    w_fb  = tf.Variable(np.random.randn(3, 3, 1, ch_q)    * 0.01, dtype=tf.float32)
    w_o   = tf.Variable(np.random.randn(ch_q, 8)          * 0.01, dtype=tf.float32)

    # initial conv layer over image+reward prior
    h = conv2d_flipkernel(X, w0, name="h0") + bias

    r = conv2d_flipkernel(h, w1, name="r")
    q = conv2d_flipkernel(r, w, name="q")
    v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    for i in range(0, k-1):
        rv = tf.concat([r, v], 3)
        wwfb = tf.concat([w, w_fb], 2)
        q = conv2d_flipkernel(rv, wwfb, name="q")
        v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    # do one last convolution
    q = conv2d_flipkernel(tf.concat([r, v], 3),
                          tf.concat([w, w_fb], 2), name="q")

    # CHANGE TO THEANO ORDERING
    # Since we are selecting over channels, it becomes easier to work with
    # the tensor when it is in NCHW format vs NHWC
    q = tf.transpose(q, perm=[0, 3, 1, 2])

    # Select the conv-net channels at the state position (S1,S2).
    # This intuitively corresponds to each channel representing an action, and the convnet the Q function.
    # The tricky thing is we want to select the same (S1,S2) position *for each* channel and for each sample
    # TODO: performance can be improved here by substituting expensive
    #       transpose calls with better indexing for gather_nd
    bs = tf.shape(q)[0]
    rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
    ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
    ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
    idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
    q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in, name="q_out")

    # add logits
    logits = tf.matmul(q_out, w_o)
    # softmax output weights
    output = tf.nn.softmax(logits, name="output")
    return logits, output
