import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
import scipy.sparse as sparse

from matplotlib import pyplot as plt
import networkx as nx

# helper methods to print nice table (taken from CGT code)
def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, float): rep = "%g"%x
    else: rep = str(x)
    return " "*(l - len(rep)) + rep

def fmt_row(width, row):
    out = " | ".join(fmt_item(x, width) for x in row)
    return out

def flipkernel(kern):
    return kern[(slice(None, None, -1),) * 2 + (slice(None), slice(None))]

def conv2d_flipkernel(x, k, name=None):
    return tf.nn.conv2d(x, flipkernel(k), name=name,
                        strides=(1, 1, 1, 1), padding='SAME')

def in_bound(x , y, width, height):
    if (x >= 0 and x < width and 
        y >= 0 and y < height ):
        return True
    else:
        return False

def adjecent_matrix(width, height):
    w0 = np.zeros([width * height, width * height])
    w1 = np.zeros([width * height, width * height])
    w2 = np.zeros([width * height, width * height])
    w3 = np.zeros([width * height, width * height])
    w4 = np.zeros([width * height, width * height])
    w5 = np.zeros([width * height, width * height])
    w6 = np.zeros([width * height, width * height])
    w7 = np.zeros([width * height, width * height])
    w8 = np.zeros([width * height, width * height])
    for i in range(width * height):
        x = i % width
        y = i / height
        if in_bound(x-1, y-1, width, height):
            w0[i][i- width - 1] = 1

        if in_bound(x, y-1, width, height):
            w1[i][i- width] = 1

        if in_bound(x+1, y-1, width, height):
            w2[i][i- width + 1] = 1

        if in_bound(x-1, y, width, height):
            w3[i][i - 1] = 1

        w4[i][i] = 1

        if in_bound(x+1, y, width, height):
            w5[i][i + 1] = 1

        if in_bound(x-1, y+1, width, height):
            w6[i][i+ width - 1] = 1

        if in_bound(x, y+1, width, height):
            w7[i][i+ width] = 1

        if in_bound(x+1, y+1, width, height):
            w8[i][i+ width + 1] = 1

    return np.array([w0, w1, w2, w3, w4, w5, w6, w7, w8])

def sparse_rprn(w):
    w_indx = np.stack([w.row, w.col])
    w_indx = np.transpose(w_indx)
    return [w_indx, w.data]

def adjecent_sparse(width, height):
    w0 = np.zeros([width * height, width * height])
    w1 = np.zeros([width * height, width * height])
    w2 = np.zeros([width * height, width * height])
    w3 = np.zeros([width * height, width * height])
    w4 = np.zeros([width * height, width * height])
    w5 = np.zeros([width * height, width * height])
    w6 = np.zeros([width * height, width * height])
    w7 = np.zeros([width * height, width * height])
    w8 = np.zeros([width * height, width * height])
    for i in range(width * height):
        x = i % width
        y = i / height
        if in_bound(x-1, y-1, width, height):
            w0[i][i- width - 1] = 1

        if in_bound(x, y-1, width, height):
            w1[i][i- width] = 1

        if in_bound(x+1, y-1, width, height):
            w2[i][i- width + 1] = 1

        if in_bound(x-1, y, width, height):
            w3[i][i - 1] = 1

        w4[i][i] = 1

        if in_bound(x+1, y, width, height):
            w5[i][i + 1] = 1

        if in_bound(x-1, y+1, width, height):
            w6[i][i+ width - 1] = 1

        if in_bound(x, y+1, width, height):
            w7[i][i+ width] = 1

        if in_bound(x+1, y+1, width, height):
            w8[i][i+ width + 1] = 1
    w = []
    w.append(sparse.coo_matrix(w0))
    w.append(sparse.coo_matrix(w1))
    w.append(sparse.coo_matrix(w2))
    w.append(sparse.coo_matrix(w3))
    w.append(sparse.coo_matrix(w4))
    w.append(sparse.coo_matrix(w5))
    w.append(sparse.coo_matrix(w6))
    w.append(sparse.coo_matrix(w7))
    w.append(sparse.coo_matrix(w8))
    w_return = []
    for i in range(len(w)):
        w_return.append(sparse_rprn(w[i]))

    return w_return

def adjecent_2DGridWorld(graph, width, height):
    w = np.zeros([width * height, width * height])
    for i in range(width * height):
        x = i % width
        y = i / height
        if (graph[x, y] != 1 ):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
            if in_bound(x-1, y-1, width, height):
                if (graph[x-1, y-1,] != 1):
                    w[i][i- width - 1] = 1

            if in_bound(x, y-1, width, height):
                if (graph[x, y-1] != 1):
                    w[i][i- width] = 1

            if in_bound(x+1, y-1, width, height):
                if (graph[x+1, y-1] != 1):
                    w[i][i- width + 1] = 1

            if in_bound(x-1, y, width, height):
                if (graph[x-1, y] != 1):
                    w[i][i - 1] = 1

            # w[i][i] = 1

            if in_bound(x+1, y, width, height):
                if (graph[x+1, y] != 1):
                    w[i][i + 1] = 1

            if in_bound(x-1, y+1, width, height):
                if (graph[x-1, y+1] != 1):
                    w[i][i+ width - 1] = 1

            if in_bound(x, y+1, width, height):
                if (graph[x, y+1] != 1):
                    w[i][i+ width] = 1

            if in_bound(x+1, y+1, width, height):
                if (graph[x+1, y+1] != 1):
                    w[i][i+ width + 1] = 1

    return w


def getPos(width, height):
    w = np.zeros([width * height, 2])
    for i in range(width * height):
        w[i][0] = i%width;
        w[i][1] = i/width;
    return w

def astar_len(graph, width, height, startx, starty, targetx, targety):
    adj = adjecent_2DGridWorld(graph, width, height)
    G = nx.from_numpy_matrix(adj)
    return nx.astar_path_length(G, starty*width+startx, targety*width+targetx)


def nx_plot(adj, pos, value):
    # input: adjacent matrix, position, value map
    label = np.arange(len(pos))
    G=nx.from_numpy_matrix(adj)

    nx.draw_networkx_nodes(G, pos, node_color = value)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)
    plt.ion()
    plt.show()


def nx_group(adj, pos, value):
    for i in range(10):
        plt.figure()
        nx_plot(adj, pos, value[i,:,0])

def extract_label(theta_matrix, start_pos, label_pos, discrete=True):
    if discrete :
        labels = []
        for i in range(len(start_pos)):
            label = []
            tmp = theta_matrix[i].toarray()[start_pos[i], label_pos[i]]
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
            labels.append(theta_matrix[i].toarray()[start_pos[i], label_pos[i]])
        labels = np.array(labels)
    return labels

# def extract_pred(output, config):

#     if config.discrete is True:
#         o_ = output.
#     else:
#         o_ = output.reshape(-1, config.statebatchsize)

#     pred = []
#     for m in range(config.batchsize):
#       buf = theta_train[m, start_train[m,:,0], :]
#       for n in range(config.statebatchsize):
#         pred.append(np.argmin(np.absolute(o_[m,n] - buf[n,buf[n].nonzero()])))
#     pred = np.array(pred)
#     pred_label = label_train[i:j].reshape(-1)
#     e_ = (np.sum(pred != pred_label))/(1.0*batch_size*config.statebatchsize)
