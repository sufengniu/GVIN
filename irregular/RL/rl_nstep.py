import time
import numpy as np
import tensorflow as tf
from data import *
from model import *
from utils import *
from tqdm import tqdm
import random
from tensorflow.python.ops import control_flow_ops
from scipy.spatial.distance import pdist, squareform

import networkx as nx

def astar_len(adj, start, target):
    G = nx.from_numpy_matrix(adj)
    return nx.astar_path_length(G, start, target)

# Data
tf.app.flags.DEFINE_string('input',           '../../data/irregular10.mat', 'Path to data')
tf.app.flags.DEFINE_integer('nodesize',       10,                    'Size of input image')
# Parameters
tf.app.flags.DEFINE_float('lr',               0.001,                  'Learning rate for RMSProp')
tf.app.flags.DEFINE_string('epsilon',         0.05,                    'Epsilon value for E-greedy')
tf.app.flags.DEFINE_string('GAMMA',           0.99,                     'Gamma value')
tf.app.flags.DEFINE_integer('epochs',         30,                     'Maximum epochs to train for')
tf.app.flags.DEFINE_integer('k',              15,                     'Number of value iterations')
tf.app.flags.DEFINE_integer('ka',             8,                      'Number of reference direction')
tf.app.flags.DEFINE_integer('t',              100,                    'kernel power')
tf.app.flags.DEFINE_integer('ch_i',           1,                      'Channels in input layer')
tf.app.flags.DEFINE_integer('ch_h',           150,                    'Channels in initial hidden layer')
tf.app.flags.DEFINE_integer('ch_q',           10,                     'Channels in q layer (~actions)')
tf.app.flags.DEFINE_integer('batchsize',      12,                     'Batch size')
tf.app.flags.DEFINE_integer('statebatchsize', 1,                     'Number of state inputs for each sample (real number, technically is k+1)')
# Misc.
tf.app.flags.DEFINE_integer('seed',           0,                      'Random seed for numpy')
tf.app.flags.DEFINE_integer('display_step',   1,                      'Print summary output every n epochs')
tf.app.flags.DEFINE_boolean('preload',        True,                  'preload the theta matrix')
tf.app.flags.DEFINE_boolean('dist',           True,                   'geo distance measure')
tf.app.flags.DEFINE_string('logdir',          '/tmp/vintf/',          'Directory to store tensorboard summary')


config = tf.app.flags.FLAGS

np.random.seed(config.seed)

# Symbolic input image tensor where typically first channel is image, second is the reward prior
X  = tf.placeholder(tf.float32, name="X",  shape=[None, config.nodesize])
Adj = tf.placeholder(tf.float32, name="Adj", shape=[None, config.nodesize, config.nodesize])
Sup = tf.placeholder(tf.float32, name="Sup", shape=[None, config.nodesize, config.nodesize])
Coord = tf.placeholder(tf.float32, name="Coord", shape=[None, config.nodesize, config.nodesize, 8])
Dist = tf.placeholder(tf.float32, name="Dist", shape=[None, config.nodesize, config.nodesize])
# Symbolic input batches of node positions
S = tf.placeholder(tf.int32,   name="S", shape=[None, config.statebatchsize])
Trainable = tf.placeholder(tf.bool, name="Trainable")
y  = tf.placeholder(tf.int32,   name="y",  shape=[None])

y_t = tf.placeholder(tf.float32, name="y_t", shape=[None])

t_1 = tf.placeholder(tf.float32, name="t_1", shape=[None])


# Construct model (Value Iteration Network)
q_net = ir_Block_coord("q", X, Adj, Dist, Sup, Coord, S,Trainable, config)

logits, nn = q_net.build_model()
  

# Variables for the loss fucntion
arg_idx = tf.cast(tf.argmax(logits, -1), tf.int32)
t = tf.gather(logits[0], arg_idx)

# Define the loss function and optimizer
loss = tf.squared_difference(t, t_1)



optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)##.minimize(loss)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=config.lr, epsilon=1e-6, centered=True)

vars_ = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, vars_), 2.0)
optimize = optimizer.apply_gradients(zip(grads, vars_))

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Preprocess the irregular graph data
data = process_irregular_data(input=config.input, preload=config.preload)


# Environment to return rewards. penalties are proportional to distances traversed
def env_ir(prev_pos, cur_pos, dist, num_steps, max_steps):
  if cur_pos == 0:
    return 1.0, True
  else:
    return -5 * dist[prev_pos,cur_pos], False
    # return -0.5, False


# Load data
adj_train, sup_train = data["adj_train"], data["support_train"]
adj_train = np.array([adj_train[i].toarray() for i in range(adj_train.shape[0])]) # adjacent matrix
sup_train = np.array([sup_train[i].toarray() for i in range(adj_train.shape[0])]) # normalization term eq. 7 in 6.3
coord_train = data["coord_train"] # node coordinates/embedding
goal_map_train = data["goal_map_train"] # goal map (vector)
start_train = data["start_train"] - 1 # where agent start
label_train = (data["label_train"] - 1).reshape(-1) # next action
degree_train = data["degree_train"] # degree matrix
geodesic_train = data["dist_train"].astype(np.float32)  # graph geodesic distance, for measuerment purpose
emb_train = coord_matrix(coord_train, data["coord_matrix_train"], config) # neural network input eq. 7 in 6.3

adj_test, sup_test = data["adj_test"], data["support_test"] 
adj_test = np.array([adj_test[i].toarray() for i in range(adj_test.shape[0])]) # adjacent matrix
sup_test = np.array([sup_test[i].toarray() for i in range(adj_test.shape[0])]) # normalization term eq. 7 in 6.3
coord_test = data["coord_test"] # node coordinates/embedding
goal_map_test = data["goal_map_test"] # goal map (vector)
start_test = data["start_test"] -1 # where agent start
label_test = (data["label_test"] - 1).reshape(-1) # next action
degree_test = data["degree_test"]  # degree matrix
dist_test = data["dist_test"] # graph geodesic distance, for measuerment purpose
emb_test = coord_matrix(coord_test, data["coord_matrix_test"], config) # normalization term eq. 7 in 6.3

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
        device_count = {'GPU': 0}
    )

# Launch the graph
saver = tf.train.Saver()
sess = tf.Session(config=SessConfig)
sess.run(init)


# Run the success rate on testing set
def test():
  totalreward = 0
  totalDiff = 0
  success = 0
  total = 0
  for i in range(adj_test.shape[0]):
    for j in range(5):
      total += 1
      current_pos = start_test[i][j][0]
      terminate = False
      step = 0
      agent_path = 0
      shortest_step = geodesic_test[i][current_pos][0]+5
      PLen = astar_len(euclidean_test[i], current_pos, 0)
      reward = 0
      ns = []
      while (terminate == False):
        step += 1
        previous = current_pos.copy()
        fd = {X: goal_map_test[i].reshape(1,-1), 
        Adj: adj_test[i].reshape(1, config.nodesize, config.nodesize), 
        Sup: sup_test[i].reshape(1,config.nodesize, config.nodesize), 
        Dist: euclidean_test[i].reshape(1,config.nodesize, config.nodesize), 
        Coord: emb_test[i].reshape(1,config.nodesize, config.nodesize, 8),
        S: np.array(current_pos).reshape(1,-1), Trainable:False}
        next_step = sess.run(logits, fd)
        current_pos = np.argmax(next_step)
        current_PLen = euclidean_test[i][previous, current_pos]
        agent_path += current_PLen
        ns.append(current_pos)

        if(step > shortest_step):
          terminate = True
          # reward -= 0.1*current_PLen
        elif current_pos == 0:
          terminate = True
          success += 1
          reward += 1
          totalDiff += agent_path - PLen
          # print("succ")
          print("Step: " + str(i) + ", Current_Succ_Rate: " + str(success/float(total)) + " Avg Difference: " + str(totalDiff/float(success)) + " Avg Reward: " + str(totalreward/float(total)))

        else:
          reward -= 0.1*current_PLen
      totalreward += reward
    



print("----- t: ", config.t, " -----")
print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))
saver = tf.train.Saver()
t_variable = tf.trainable_variables()


# Hyperparameters for episodic q learning
random_decay = 0.9995
eps = config.epsilon
step_buff = np.zeros(adj_train.shape[0]*5)




# begin epochs
total_steps = 0
accumulated_rewards = 0
for epoch in range(200):
  success = 0
  total = 0
   
  for i in range(adj_test.shape[0]):
    for j in range(5):
      reward = 0

      y_t = 0
      error = []
      total += 1
      s_t = np.random.randint(1,config.nodesize)
      terminate = False
      step = 0
      shortest_step = geodesic_train[i][s_t][0]
      error.append([s_t, shortest_step, step])
      iter_ = 0
      ri = []
      b_s_t = []
      b_s_t1 = []
      goal_maps = [] 
      adjs = []
      dists = []
      rewards = []
      sups = []
      Coords = []
      Ss = []
      while (terminate == False):
        step += 1
        total_steps += 1


        fd = {X: goal_map_train[i].reshape(1,-1), 
        Adj: adj_train[i].reshape(1, config.nodesize, config.nodesize), 
        Dist: euclidean_train[i].reshape(1, config.nodesize, config.nodesize),
        Sup: sup_train[i].reshape(1,config.nodesize, config.nodesize), 
        Coord: emb_train[i].reshape(1,config.nodesize, config.nodesize, 8),
        S: np.array(s_t).reshape(1,-1), Trainable:True}
        next_step = sess.run(logits, fd)

        # possible postion
        poss_steps = np.nonzero(next_step)

        # epislon greedy
        nA = config.nodesize
        A = np.ones(nA, dtype=float) * eps / nA
        best_action = np.argmax(next_step)
        A[best_action] += (1.0 - eps)
        s_t1 = np.random.choice(np.arange(len(A)), p=A)


        q = next_step[0][s_t1]

        reward, done = env_ir(s_t, s_t1, euclidean_train[i], step, step_buff[i*5 + j])

        accumulated_rewards += reward 
        # store states for episodic updates
        goal_maps.append(goal_map_train[i])
        adjs.append(adj_train[i])
        dists.append(euclidean_train[i])
        sups.append(sup_train[i])
        Coords.append(emb_train[i])
        rewards.append(reward)
        Ss.append(s_t)

        # terminate if current step is 5 steps more than shortest path
        if(step > shortest_step+5):
          terminate = True
          if step_buff[i*5 +j] == 0:
            step_buff[i*5 +j] = shortest_step + 3
        # 0 is the target position
        elif s_t1 == 0:
          terminate = True
          success += 1
          if step_buff[i*5 +j] == 0 or step_buff[i*5 +j] > step:
            step_buff[i*5 +j] = step
        s_t = s_t1

      rt = np.zeros(step)
      for m in reversed(range(step)):
        if m == step - 1:
          rt[m] = rewards[m] + 0
        else:
          rt[m] = rewards[m] + config.GAMMA * rt[m+1]

      fd = {X: np.array(goal_maps), Adj: np.array(adjs), Dist: np.array(dists), Sup: np.array(sups), Coord: np.array(Coords),
            S: np.array(Ss).reshape(-1,1), t_1: rt, Trainable:True}

      _, l = sess.run([optimize, loss], fd)

    if (np.mod(i, 5)==0):
        eps = eps * random_decay
  if np.mod(epoch, 5)==0 :
    # test()
    print("Epoch: " + str(epoch) + " Step: " + str(total_steps) + ", Current_Acc: " + str(success/float(total))
      + " Decay: " + str(eps) + " Accumulated Rewards " + str(accumulated_rewards/total_steps))

test()
  # saver.save(sess, "./model/model_RL_10_dropout.ckpt")
