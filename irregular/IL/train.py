import time
import numpy as np
import tensorflow as tf
from data import *
from model import *
from utils import *
from tqdm import tqdm

from scipy.spatial.distance import pdist, squareform

# Data
tf.app.flags.DEFINE_string('input',           '../../data/irregular100.mat', 'Path to data')
tf.app.flags.DEFINE_integer('nodesize',       100,                    'Size of input image')
# Parameters
tf.app.flags.DEFINE_float('lr',               0.001,                  'Learning rate for RMSProp')
tf.app.flags.DEFINE_integer('epochs',         50,                     'Maximum epochs to train for')
tf.app.flags.DEFINE_integer('k',              30,                     'Number of value iterations')
tf.app.flags.DEFINE_integer('ka',             8,                      'Number of reference direction')
tf.app.flags.DEFINE_integer('t',              20,                    'kernel power')
tf.app.flags.DEFINE_integer('ch_i',           1,                      'Channels in input layer')
tf.app.flags.DEFINE_integer('ch_h',           150,                    'Channels in initial hidden layer')
tf.app.flags.DEFINE_integer('ch_q',           10,                     'Channels in q layer (~actions)')
tf.app.flags.DEFINE_integer('batchsize',      12,                     'Batch size')
tf.app.flags.DEFINE_integer('statebatchsize', 5,                     'Number of state inputs for each sample (real number, technically is k+1)')
# Misc.
tf.app.flags.DEFINE_integer('seed',           0,                      'Random seed for numpy')
tf.app.flags.DEFINE_integer('display_step',   1,                      'Print summary output every n epochs')
tf.app.flags.DEFINE_boolean('preload',        True,                  'preload the theta matrix')
tf.app.flags.DEFINE_boolean('log',            False,                  'Enable for tensorboard summary')
tf.app.flags.DEFINE_boolean('dist',           False,                   'geo distance measure')
tf.app.flags.DEFINE_string('logdir',          '/tmp/vintf/',          'Directory to store tensorboard summary')

config = tf.app.flags.FLAGS

np.random.seed(config.seed)

# symbolic input image tensor where typically first channel is image, second is the reward prior
X  = tf.placeholder(tf.float32, name="X",  shape=[None, config.nodesize])
Adj = tf.placeholder(tf.float32, name="Adj", shape=[None, config.nodesize, config.nodesize])
Sup = tf.placeholder(tf.float32, name="Adj", shape=[None, config.nodesize, config.nodesize])
Coord = tf.placeholder(tf.float32, name="Coord", shape=[None, config.nodesize, config.nodesize, 8])
Dist = tf.placeholder(tf.float32, name="Dist", shape=[None, config.nodesize, config.nodesize])
# Coord = tf.placeholder(tf.float32, name="Coord", shape=[None, config.nodesize, config.nodesize, 4])
# symbolic input batches of node positions
S = tf.placeholder(tf.int32,   name="S", shape=[None, config.statebatchsize])

y  = tf.placeholder(tf.int32,   name="y",  shape=[None])
lr = tf.placeholder(tf.float32,   name="lr")

# Construct model (Value Iteration Network)
val_map, logits, nn = ir_Block(X, Adj, Dist, Sup, Coord, S, config)

# Define loss and optimizer
y_ = tf.cast(y, tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=y_, name='cross_entropy')
cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
tf.add_to_collection('losses', cross_entropy_mean)

cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, epsilon=1e-6, centered=True)
# optimizer = tf.train.AdamOptimizer(learning_rate=lr)
vars_ = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, vars_), 2.0)
optimize = optimizer.apply_gradients(zip(grads, vars_))

# Test model & calculate accuracy
cp = tf.cast(tf.argmax(nn, 1), tf.int32)
err = tf.reduce_mean(tf.cast(tf.not_equal(cp, y), dtype=tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

data = process_irregular_data(input=config.input, preload=config.preload)

# training set
adj_train, sup_train = data["adj_train"], data["support_train"]
adj_train = np.array([adj_train[i].toarray() for i in range(adj_train.shape[0])]) # adjacent matrix
sup_train = np.array([sup_train[i].toarray() for i in range(adj_train.shape[0])]) # normalization term eq. 7 in 6.3
coord_train = data["coord_train"] # node coordinates/embedding
goal_map_train = data["goal_map_train"] # goal map (vector)
start_train = data["start_train"] - 1  # where agent start
label_train = (data["label_train"] - 1).reshape(-1) # next action
degree_train = data["degree_train"] # degree matrix
geodesic_train = data["dist_train"].astype(np.float32) # graph geodesic distance, for measuerment purpose
emb_train = coord_matrix(coord_train, data["coord_matrix_train"], config) # neural network input eq. 7 in 6.3

# testing set
adj_test, sup_test = data["adj_test"], data["support_test"]
adj_test = np.array([adj_test[i].toarray() for i in range(adj_test.shape[0])]) # adjacent matrix
sup_test = np.array([sup_test[i].toarray() for i in range(adj_test.shape[0])]) # normalization term eq. 7 in 6.3
coord_test = data["coord_test"] # node coordinates/embedding
goal_map_test = data["goal_map_test"] # goal map (vector)
start_test = data["start_test"] -1  # where agent start
label_test = (data["label_test"] - 1).reshape(-1) # next action
degree_test = data["degree_test"] # degree matrix
geodesic_test = data["dist_test"].astype(np.float32) # graph geodesic distance, for measuerment purpose
emb_test = coord_matrix(coord_test, data["coord_matrix_test"], config) # neural network input eq. 7 in 6.3

# graph euclidean distance, for measuerment purpose
euclidean_train = []
for i in range(adj_train.shape[0]):
  euclidean_train.append(squareform(pdist(coord_train[i]))*adj_train[i])
euclidean_train = np.array(euclidean_train)
euclidean_test = []
for i in range(adj_test.shape[0]):
  euclidean_test.append(squareform(pdist(coord_test[i]))*adj_test[i])
euclidean_test = np.array(euclidean_test)



SessConfig = tf.ConfigProto(
  # device_count = {'GPU': 0}
  )
# Launch the graph
sess = tf.Session(config=SessConfig)
if config.log:
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)
  summary_op = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter(config.logdir, sess.graph)
sess.run(init)
# saver.restore(sess, "./model/model_10_coord.ckpt")


def test():
  num_batches = int(goal_map_test.shape[0]/batch_size)
  avg_err = 0.0
  for i in range(0, goal_map_test.shape[0], batch_size):
    j = i + batch_size
    if j <= goal_map_test.shape[0]:
      # Run optimization op (backprop) and cost op (to get loss value)
      fd = {X: goal_map_test[i:j],
      Adj: adj_test[i:j], 
      Dist: euclidean_test[i:j], 
      Sup: sup_test[i:j],
      Coord: emb_test[i:j],
      S: np.squeeze(start_test[i:j]), 
      y:label_test[i * config.statebatchsize:j * config.statebatchsize]}

      if config.dist is False:
        c_, e_ = sess.run([cost, err], feed_dict=fd)
      else:
        c_, o_ = sess.run([cost, nn], feed_dict=fd)
        pred = np.argmax(o_,axis=1).reshape(batch_size, -1)
        pred_label = label_test[i * config.statebatchsize:j * config.statebatchsize].reshape(batch_size, -1)
        p_dist = np.array([geodesic_test[i:j][m,0,pred[m,:]] for m in range(batch_size)]).reshape(-1)
        pl_dist = np.array([geodesic_test[i:j][m,0,pred_label[m,:]] for m in range(batch_size)]).reshape(-1)
        e_ = (np.sum(p_dist > pl_dist))/(1.0*batch_size*config.statebatchsize)
      avg_err += e_

  print("Testing Accuracy: {}%".format(100 * (1 - avg_err/num_batches)))


batch_size    = config.batchsize
print("----- t: ", config.t, " -----")
print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))
saver = tf.train.Saver()
t_variable = tf.trainable_variables()
lr_fed = 0.002
for epoch in range(int(config.epochs)):
  tstart = time.time()
  avg_err, avg_cost = 0.0, 0.0
  num_batches = int(goal_map_train.shape[0]/batch_size)

  if epoch == 20:
    lr_fed = 0.001
  elif epoch == 30:
    lr_fed = 0.0005
  elif epoch == 40:
    lr_fed = 0.0002

  # Loop over all batches
  for i in tqdm(range(0, goal_map_train.shape[0], batch_size)):
    j = i + batch_size
    if j <= goal_map_train.shape[0]:
      # Run optimization op (backprop) and cost op (to get loss value)
      fd = {X: goal_map_train[i:j], Adj: adj_train[i:j], Sup: sup_train[i:j], Coord: emb_train[i:j],
          Dist: euclidean_train[i:j], lr: lr_fed,
          S: np.squeeze(start_train[i:j]), y: label_train[i * config.statebatchsize:j * config.statebatchsize]}

      if config.dist is False:
        _, c_, e_ = sess.run([optimize, cost, err], feed_dict=fd)
      else:
        _, c_, o_ = sess.run([optimize, cost, nn], feed_dict=fd)
        pred = np.argmax(o_,axis=1).reshape(batch_size, -1)
        pred_label = label_train[i * config.statebatchsize:j * config.statebatchsize].reshape(batch_size, -1)
        p_dist = np.array([geodesic_train[i:j][m,0,pred[m,:]] for m in range(batch_size)]).reshape(-1)
        pl_dist = np.array([geodesic_train[i:j][m,0,pred_label[m,:]] for m in range(batch_size)]).reshape(-1)
        e_ = (np.sum(p_dist != pl_dist))/(1.0*batch_size*config.statebatchsize)

      avg_err += e_
      avg_cost += c_

      # if c_ > 50000.0:
      #   break

  # Display logs per epoch step
  if epoch % config.display_step == 0:
    elapsed = time.time() - tstart
    print(fmt_row(10, [epoch, avg_cost/num_batches, avg_err/num_batches, elapsed]))
  if config.log:
    summary = tf.Summary()
    summary.ParseFromString(sess.run(summary_op))
    summary.value.add(tag='Average error', simple_value=float(avg_err/num_batches))
    summary.value.add(tag='Average cost', simple_value=float(avg_cost/num_batches))
    summary_writer.add_summary(summary, epoch)
saver.save(sess, "./model/model_10.ckpt")

print("Finished training!")

test()

