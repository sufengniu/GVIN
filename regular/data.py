import numpy as np
import scipy.io as sio
import scipy
from scipy.sparse import *
from scipy import *
from tqdm import tqdm

def process_gridworld_data(input, imsize):
    # run training from input matlab data file, and save test data prediction in output file
    # load data from Matlab file, including
    # im_data: flattened images
    # state_data: concatenated one-hot vectors for each state variable
    # state_xy_data: state variable (x,y position)
    # label_data: one-hot vector for action (state difference)
    im_size=[imsize, imsize]
    matlab_data = sio.loadmat(input)
    im_data = matlab_data["batch_im_data"]
    im_data = (im_data - 1)/255  # obstacles = 1, free zone = 0
    value_data = matlab_data["batch_value_data"]
    state1_data = matlab_data["state_x_data"]
    state2_data = matlab_data["state_y_data"]
    label_data = matlab_data["batch_label_data"]
    ydata = label_data.astype('int8')
    Xim_data = im_data.astype('float32')
    Xim_data = Xim_data.reshape(-1, 1, im_size[0], im_size[1])
    Xval_data = value_data.astype('float32')
    Xval_data = Xval_data.reshape(-1, 1, im_size[0], im_size[1])
    Xdata = np.append(Xim_data, Xval_data, axis=1)
    # Need to transpose because Theano is NCHW, while TensorFlow is NHWC
    Xdata = np.transpose(Xdata,  (0, 2, 3, 1))
    S1data = state1_data.astype('int8')
    S2data = state2_data.astype('int8')

    all_training_samples = int(6/7.0*Xdata.shape[0])
    training_samples = all_training_samples
    Xtrain = Xdata[0:training_samples]
    S1train = S1data[0:training_samples]
    S2train = S2data[0:training_samples]
    ytrain = ydata[0:training_samples]

    Xtest = Xdata[all_training_samples:]
    S1test = S1data[all_training_samples:]
    S2test = S2data[all_training_samples:]
    ytest = ydata[all_training_samples:]
    ytest = ytest.flatten()

    sortinds = np.random.permutation(training_samples)
    Xtrain = Xtrain[sortinds]
    S1train = S1train[sortinds]
    S2train = S2train[sortinds]
    ytrain = ytrain[sortinds]
    ytrain = ytrain.flatten()
    return Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest

# def angle(p):
#     ang = np.arctan2(*p[::-1])
#     return ang % (2 * np.pi)

def angle(p):
    ang = np.arctan2(p[:,1], p[:,0])
    return ang % (2 * np.pi)

def theta_matrix(coord, adj, preload=True, train=True):
    print "creating adjacent theta matrix ..."
    if preload is True:
        if train is True:
            theta_matrix = np.load('../data/theta_matrix_train_n_100.npy')
        else:
            theta_matrix = np.load('../data/theta_matrix_test_n_100.npy')
    else:
        theta_matrix = []
        for i in tqdm(range(coord.shape[0])):
            for j in range(coord.shape[1]):
                theta_row = angle(coord[i,adj[i][j].nonzero()[1],:] - coord[i,j,:])
                col_indice = adj[i][j].nonzero()[1]
                row_indice = (np.zeros(col_indice.shape[0])).astype(int32)
                if j == 0:
                    theta_matrix_tmp = csc_matrix((theta_row, (row_indice, col_indice)), shape=(1,coord.shape[1]))
                else:
                    theta_matrix_tmp = scipy.sparse.vstack((theta_matrix_tmp, csc_matrix((theta_row, (row_indice, col_indice)), shape=(1,coord.shape[1]))))
            theta_matrix.append(theta_matrix_tmp)
        theta_matrix = np.array(theta_matrix)
    return theta_matrix

def coord_matrix(coord):
    coord_matrix = []
    for i in range(coord.shape[0]):
        coord_tmp = []
        for j in range(coord.shape[1]):
            coord_tmp.append(coord[i,j,:] - coord[i,:,:])
        coord_matrix.append(np.array(coord_tmp))
    return np.array(coord_matrix)

def goal_gen(coord):
    # goal_coord = coord[:, 0, 0]
    # idxs = coord[:,:,0].argsort()
    # goal_permuted_pos = np.where(idxs==0)[1]
    goal_map = np.zeros([coord.shape[0], coord.shape[1]])
    # coord_permuted = []
    # for i in range(coord.shape[0]):
        # coord_permuted.append(coord[i,:,:][idxs[i]])
    goal_map[:,0] = 500
    # coord_permuted = np.array(coord_permuted)
    return goal_map

def extract_label(theta_matrix, start_pos, label_pos, discrete=True):
    if discrete :
        labels = []
        for i in range(len(start_pos)):
            label = []
            tmp = theta_matrix[i].toarray()[start_pos[i]-1, label_pos[i]-1]
            for j in range(len(tmp)):
                if tmp[j] <= np.pi/8 or tmp[j] > 15*np.pi/8:
                    label.append(0)
                elif tmp[j] <= 3*np.pi/8 and tmp[j] > np.pi/8:
                    label.append(1)
                elif tmp[j] <= 5*np.pi/8 and tmp[j] > 3*np.pi/8:
                    label.append(2)          
                elif tmp[j] <= 7*np.pi/8 and tmp[j] > 5*np.pi/8:
                    label.append(3)
                elif tmp[j] <= 9*np.pi/8 and tmp[j] > 7*np.pi/8:
                    label.append(4)
                elif tmp[j] <= 11*np.pi/8 and tmp[j] > 9*np.pi/8:
                    label.append(5)
                elif tmp[j] <= 13*np.pi/8 and tmp[j] > 11*np.pi/8:
                    label.append(6)
                elif tmp[j] <= 15*np.pi/8 and tmp[j] > 13*np.pi/8:
                    label.append(7)
            labels.append(label)
        labels = np.array(labels) 
    else :
        labels = []
        for i in range(len(start_pos)):
            labels.append(theta_matrix[i].toarray()[start_pos[i]-1, label_pos[i]-1])
        labels = np.array(labels)
    return labels

def degree_matrix(adj):
    output = []
    for i in range(len(adj)):
        adj_dense = adj[i].toarray()
        I = np.zeros([adj_dense.shape[0], adj_dense.shape[1]])
        np.fill_diagonal(I, 1.0)
        adj_hat = adj_dense + I
        D_diag = np.power(np.sum(adj_hat, axis=1), -0.5)
        D = np.zeros([adj_dense.shape[0], adj_dense.shape[1]])
        np.fill_diagonal(D, D_diag)
        output.append(D)
    return np.array(output)

def process_irregular_data(input, preload):
    print "loading input data..."
    matlab_data = sio.loadmat(input)
    G = matlab_data['G']
    adj_data, coord_data, goal_pos, start_pos, label_pos, dist = [], [], [], [], [], []
    for i in range(G.shape[1]):
        adj_data.append(G[0,i][0])
        coord_data.append(G[0,i][1])
        goal_pos.append(G[0,i][2])
        start_pos.append(G[0,i][3])
        label_pos.append(G[0,i][4])
        dist.append(G[0,i][5])
    adj_data = np.array(adj_data)
    coord_data = np.array(coord_data).astype('float32')
    goal_pos = np.array(goal_pos).astype('int8')
    start_pos = np.array(start_pos).astype('int8')
    label_pos = np.array(label_pos).astype('int8')
    dist_data = np.array(dist).astype('int8')

    all_training_samples = int(6/7.0*G.shape[1])
    training_samples = all_training_samples
    data = {}
    data["adj_train"] = adj_data[0:training_samples]
    data["coord_train"] = coord_data[0:training_samples]
    data["theta_train"] = theta_matrix(data["coord_train"], data["adj_train"], preload, train=True)
    data["goal_map_train"] = goal_gen(data["coord_train"])
    data["start_train"] = start_pos[0:training_samples]
    data["y_train"] = extract_label(data["theta_train"], data["start_train"], label_pos[0:training_samples])
    data["label_train"] = label_pos[0:training_samples]
    data["degree_train"] = degree_matrix(data["adj_train"])
    data["dist_train"] = dist_data[:training_samples]
    data["coord_matrix_train"] = coord_matrix(data["coord_train"])

    data["adj_test"] = adj_data[training_samples:]
    data["coord_test"] = coord_data[training_samples:]
    data["theta_test"] = theta_matrix(data["coord_test"], data["adj_test"], preload, train=False)
    data["goal_map_test"] = goal_gen(data["coord_test"])
    data["start_test"] = start_pos[training_samples:]
    data["y_test"] = extract_label(data["theta_test"], data["start_test"], label_pos[training_samples:])
    data["label_test"] = label_pos[training_samples:]
    data["degree_test"] = degree_matrix(data["adj_test"])
    data["dist_test"] = dist_data[training_samples:]
    data["coord_matrix_test"] = coord_matrix(data["coord_test"])
    
    return data


