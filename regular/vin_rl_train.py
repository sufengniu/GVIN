import time
import numpy as np
import tensorflow as tf
from data  import *
# from hierarchy_model import *
from vin_model import *
from utils import *

np.random.seed(0)

# Data
tf.app.flags.DEFINE_string('input',           '../data/gridworld_16.mat', 'Path to data')
tf.app.flags.DEFINE_integer('imsize',         16,                      'Size of input image')
# Parameters
tf.app.flags.DEFINE_float('lr',               0.001,                  'Learning rate for RMSProp')
tf.app.flags.DEFINE_float('epsilon',          0.3,                    'Epsilon for e-greedy')
tf.app.flags.DEFINE_float('GAMMA',            0.99,                    'discount factor')
tf.app.flags.DEFINE_integer('epochs',         20,                     'Maximum epochs to train for')
tf.app.flags.DEFINE_integer('k',              20,                     'Number of value iterations')
tf.app.flags.DEFINE_integer('ch_i',           2,                      'Channels in input layer')
tf.app.flags.DEFINE_integer('ch_h',           150,                    'Channels in initial hidden layer')
tf.app.flags.DEFINE_integer('ch_q',           10,                     'Channels in q layer (~actions)')
tf.app.flags.DEFINE_integer('batchsize',      20,                     'Batch size')
tf.app.flags.DEFINE_integer('statebatchsize', 1,                     'Number of state inputs for each sample (real number, technically is k+1)')
tf.app.flags.DEFINE_boolean('untied_weights', False,                  'Untie weights of VI network')
# Misc.
tf.app.flags.DEFINE_integer('display_step',   1,                      'Print summary output every n epochs')
tf.app.flags.DEFINE_boolean('log',            False,                  'Enable for tensorboard summary')
tf.app.flags.DEFINE_string('logdir',          '/tmp/vintf/',          'Directory to store tensorboard summary')

config = tf.app.flags.FLAGS

# symbolic input image tensor where typically first channel is image, second is the reward prior
X  = tf.placeholder(tf.float32, name="X",  shape=[None, config.imsize, config.imsize, config.ch_i])
# symbolic input batches of vertical positions
S1 = tf.placeholder(tf.int32,   name="S1", shape=[None, config.statebatchsize])
# symbolic input batches of horizontal positions
S2 = tf.placeholder(tf.int32,   name="S2", shape=[None, config.statebatchsize])

y  = tf.placeholder(tf.int32,   name="y",  shape=[None])

t_1 = tf.placeholder(tf.float32, name="t_1", shape=[None])

# Construct model (Value Iteration Network)
q_net = VI_Block("q_net", X, S1, S2, config)
target_net = VI_Block("t_net", X, S1, S2, config)
logits, nn, value_map, logits2, nn2 = q_net.build_model()
logits_t, nn_t, value_map_t, logits2_t, nn2_t = target_net.build_model()

arg_idx = tf.cast(tf.argmax(logits2, -1), tf.int32)
logits2 = tf.reshape(logits2, [-1,8])
t = tf.reduce_max(logits2, axis=1)

loss = tf.squared_difference(t, t_1)

optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
vars_ = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, vars_), 5.0)
optimize = optimizer.apply_gradients(zip(grads, vars_))

cp = tf.cast(tf.argmax(nn2, 1), tf.int32)
err = tf.reduce_mean(tf.cast(tf.not_equal(cp, y), dtype=tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver()

Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest = process_gridworld_data(input=config.input, imsize=config.imsize)

# environment to return rewards. penalties are proportional to distances traversed
def env_ir(x, y, goal_x, goal_y, canvas):
  if canvas[x, y] == 5:
    return 1.0, True, x, y
  elif (x <= 0 or x >= config.imsize-1 or y <= 0 or y >= config.imsize-1):
    return -1.0, True, 2, 2
  elif canvas[x, y] == 1:
    return -1.0, True, x, y
  else:
    return -0.05, False, x, y  #* dist_train_e[prev_pos,cur_pos],

# Visualize the walking
def action2cord(a):
  """
  input  : action 0~7
  output : x ,y changes
  """
  # return {'0':[0,-1],'1':[0,1],'2':[1,0],'3':[-1,0],'4':[1,-1],'5':[-1,-1],'6':[1,1],'7':[-1,1]}.get(str(a),[0,0])
  return {'0':[-1,0],'1':[1,0],'2':[0,1],'3':[0,-1],'4':[-1,1],'5':[-1,-1],'6':[1,1],'7':[1,-1]}.get(str(a),[0,0])


SessConfig = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
saver = tf.train.Saver()
sess = tf.Session(config=SessConfig)
sess.run(init)

batch_size    = config.batchsize
print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))
saver = tf.train.Saver()
t_variable = tf.trainable_variables()


nA = 8 # number of actions

epsilon_start = config.epsilon
epsilon_end = 0.0001
epsilon_decay_steps = 100
epsilon = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

for epoch in range(100):
  rand_idx = np.random.permutation(np.arange(len(Xtrain)))
  domain_index = np.random.randint(Xtrain.shape[0])  # pick a domain randomly
  success = 0
  total = 0
  tot_steps = 0
  tot_rewards = 0
  avg_cost = 0

  for i, index in enumerate(rand_idx):
    total += 1
    obstacle = True

    while(obstacle):  # random pick a start point if there is obstacle, pick other one
      
      start_x = np.random.randint(1, config.imsize - 1)
      start_y = np.random.randint(1, config.imsize - 1)
      
      if (Xtrain[index, start_x, start_y, 0] == 0):
        obstacle = False

    # retrieve current position
    current_x = np.array(start_x).reshape(1,1)
    current_y = np.array(start_y).reshape(1,1)

    domain = Xtrain[index, :, :, :].reshape(1, config.imsize, config.imsize, 2)

    goal_x = np.where(domain==10.)[1][0]
    goal_y = np.where(domain==10.)[2][0]

    terminate = False
    step = 0
    maximum_step = config.imsize*2

    canvas = domain[0,:,:,0].copy()
    # canvas_orig = domain[0,:,:,0].copy()
    canvas[goal_x, goal_y] = 5

    iter_ = 0
    ri = []
    b_s_t = []
    domain_in = []

    while (terminate==False):
      iter_ += 1
      step += 1
      tot_steps += 1

      eps = epsilon[min(epoch, epsilon_decay_steps)]
      A = np.ones(nA, dtype=float) * eps / nA
      q_values = sess.run(logits2, {X: domain, S1: current_x, S2: current_y})
      best_action = np.argmax(q_values)
      A[best_action] += (1.0 - eps)
      action_idx = np.random.choice(np.arange(len(A)), p=A)

      delta_x, delta_y = action2cord(action_idx)
      next_x = current_x[0,0] + delta_x
      next_y = current_y[0,0] + delta_y

      reward, terminate, next_x, next_y = env_ir(next_x, next_y, goal_x, goal_y, canvas)

      tot_rewards += reward
      ri.append(reward)

      b_s_t.append(np.array([current_x[0,0], current_y[0,0]]))
      domain_in.append(np.squeeze(domain))

      if(next_x == goal_x and next_y == goal_y):
        success += 1
      if iter_ > maximum_step:
        terminate = True
      current_x[0,0] = next_x
      current_y[0,0] = next_y

    rt = np.zeros(iter_)
    for m in reversed(range(iter_)):
      if m == iter_ - 1:
        rt[m] = ri[m]
      else:
        rt[m] = ri[m] + config.GAMMA * rt[m+1]

    fd = {X: np.array(domain_in), S1: np.array(b_s_t)[:,:1], S2: np.array(b_s_t)[:,1:], t_1: rt}
    _, c_= sess.run([optimize, loss], feed_dict=fd)
    avg_cost += sum(c_)

    if i%100 == 0:
      print("Epoch: " + str(epoch) + " Step: " + str(i) + ", Success rate: " + str(success/float(total)) + ", Accum rewards: " + str(tot_rewards/tot_steps))
      saver.save(sess, "./model/vin_rl_16.ckpt")



# testing

rand_idx = np.random.permutation(np.arange(len(Xtest)))
domain_index = np.random.randint(Xtest.shape[0])  # pick a domain randomly
success = 0
total = 0
tot_steps = 0
tot_rewards = 0
avg_cost = 0

for i, index in enumerate(rand_idx):
  total += 1
  obstacle = True

  while(obstacle):  # random pick a start point if there is obstacle, pick other one
    
    start_x = np.random.randint(1, config.imsize - 1)
    start_y = np.random.randint(1, config.imsize - 1)
    
    if (Xtest[index, start_x, start_y, 0] == 0):
      obstacle = False

  # retrieve current position
  current_x = np.array(start_x).reshape(1,1)
  current_y = np.array(start_y).reshape(1,1)

  domain = Xtest[index, :, :, :].reshape(1, config.imsize, config.imsize, 2)

  goal_x = np.where(domain==10.)[1][0]
  goal_y = np.where(domain==10.)[2][0]

  terminate = False
  step = 0
  maximum_step = config.imsize*2

  canvas = domain[0,:,:,0].copy()
  # canvas_orig = domain[0,:,:,0].copy()
  canvas[goal_x, goal_y] = 5

  iter_ = 0
  ri = []
  b_s_t = []
  domain_in = []

  while (terminate==False):
    iter_ += 1
    step += 1
    tot_steps += 1

    A = np.zeros(nA, dtype=float)
    q_values = sess.run(logits2, {X: domain, S1: current_x, S2: current_y})
    best_action = np.argmax(q_values)
    A[best_action] = 1.0
    action_idx = np.random.choice(np.arange(len(A)), p=A)

    delta_x, delta_y = action2cord(action_idx)
    next_x = current_x[0,0] + delta_x
    next_y = current_y[0,0] + delta_y

    reward, terminate, next_x, next_y = env_ir(next_x, next_y, goal_x, goal_y, canvas)

    tot_rewards += reward
    ri.append(reward)

    b_s_t.append(np.array([current_x[0,0], current_y[0,0]]))
    domain_in.append(np.squeeze(domain))

    if(next_x == goal_x and next_y == goal_y):
      success += 1
    if iter_ > maximum_step:
      terminate = True
    current_x[0,0] = next_x
    current_y[0,0] = next_y

  print("Step: " + str(i) + ", Current_Acc: " + str(success/float(total)) + ", accum rewards: " + str(tot_rewards/tot_steps))


