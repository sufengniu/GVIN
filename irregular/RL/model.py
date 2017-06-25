import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils import conv2d_flipkernel, adjecent_matrix, adjecent_sparse

def dot(x, y, sparse=False):
    if sparse:
        return tf.sparse_tensor_dense_matmul(x, y)
    else:
        return tf.matmul(x, y)

def kernel_net_coord(coord, weight, train, config):
    x = tf.reshape(coord, [-1, 8])
    x = tf.concat([x, tf.reshape(weight, [-1,1])], axis=1)
    x = slim.fully_connected(x, 32, activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), scope="fc_1")
    x = slim.fully_connected(x, 64, activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), scope="fc_2")
    x = slim.fully_connected(x, 1, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), scope="fc_3")
    return tf.reshape(x, [-1, config.nodesize, config.nodesize])



class ir_Block_coord():

    def __init__(self, scope, X, Adj, Dist, Support, Coord, S, train, config):
        self.scope = scope
        self.X = X
        self.Adj = Adj
        self.Support = Support
        self.Dist = Dist
        self.Coord = Coord
        self.S = S
        self.config = config
        self.train = train


    def build_model(self):
        with tf.variable_scope(self.scope):
            return self.build_ir_Block_coord()


    def build_ir_Block_coord(self):
        ka   = self.config.ka
        k    = self.config.k    # Number of value iterations performed
        t    = self.config.t
        ch_i = self.config.ch_i # Channels in input layer
        ch_h = self.config.ch_h # Channels in initial hidden layer
        ch_q = self.config.ch_q # Channels in q layer (~actions)
        state_batch_size = self.config.statebatchsize # k+1 state inputs for each channel
        bs   = self.config.batchsize
        img_s = self.config.nodesize

        
        diag = np.zeros([img_s, img_s])
        np.fill_diagonal(diag, 1.0)
        DO_SHARE=None

        # Generate weights using kernel_net_coord
        P = []
        for i in range(ch_q):
            with tf.variable_scope('p_'+str(i)):
                coeff = kernel_net_coord(self.Coord, self.Dist, self.train, self.config)
                coeff = coeff * self.Support
            P.append(coeff)


        P_fb = []
        for i in range(ch_q):
            with tf.variable_scope('pb_'+str(i)):
                coeff = kernel_net_coord(self.Coord, self.Dist, self.train, self.config)
                coeff = coeff * self.Support
            P_fb.append(coeff)
        P = tf.transpose(tf.stack(P), [1,0,2,3])
        P_fb = tf.transpose(tf.stack(P_fb), [1,0,2,3])


        # Copy r for each channel
        r_ = self.X
        r_repeat = []
        for j in range(ch_q):
            r_repeat.append(r_)
        r_repeat = tf.stack(r_repeat)
        r_repeat = tf.transpose(r_repeat, [1,0,2])
        r_repeat = tf.expand_dims(r_repeat, axis=-1)


        q = tf.matmul(P, r_repeat)
        v = tf.reduce_max(q, axis=1, keep_dims=True, name="v")
        v_ = tf.reshape(v, [-1, img_s])

        # Copy v for each channel
        v_repeat = []
        for i in range(ch_q):
            v_repeat.append(v_)
        v_repeat = tf.stack(v_repeat)
        v_repeat = tf.transpose(v_repeat, [1,0,2])
        v_repeat = tf.expand_dims(v_repeat, axis=-1)

        for i in range(0, k-1):
            q1 = tf.matmul(P, r_repeat)
            q2 = tf.matmul(P_fb, v_repeat)
            q = q1 + q2
            v = tf.reduce_max(q, axis=1, keep_dims=True, name="v")
            v_ = tf.reshape(v, [-1, img_s])
            v_repeat = []
            for j in range(ch_q):
                v_repeat.append(v_)
            v_repeat = tf.stack(v_repeat)
            v_repeat = tf.transpose(v_repeat, [1,0,2])
            v_repeat = tf.expand_dims(v_repeat, axis=-1)

        q1 = tf.matmul(P, r_repeat)
        q2 = tf.matmul(P_fb, v_repeat)
        q = q1 + q2

        # Select adjcent node to generate next action probability.
        bs = tf.shape(q)[0]
        rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
        ins = tf.cast(tf.reshape(self.S, [-1]), tf.int32)
        idx_in = tf.transpose(tf.stack([ins, rprn]), [1, 0])
        v_idx = tf.gather_nd(tf.transpose(self.Adj, [2,0,1]), idx_in, name="v_out")
        v_out_rp = []
        for j in range(state_batch_size):
            v_out_rp.append(v_)
        v_out_rp = tf.stack(v_out_rp)
        v_out_rp = tf.transpose(v_out_rp, [1,0,2])
        v_out_rp = tf.reshape(v_out_rp, [-1, img_s])
        logits = tf.multiply(v_idx, v_out_rp)
        output = tf.nn.softmax(logits, name="output")
        return logits, output


