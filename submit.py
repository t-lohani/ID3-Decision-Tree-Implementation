from __future__ import division

import argparse
import csv
import pickle as pkl
import random
import collections
import numpy as np
from scipy.stats import chi2
import copy
fp = open("our_output","w")
def get_column_data_map(data, column):
    return collections.Counter([row[column] for row in data])

def get_child(node, value):
    # print "looking for value : " + str(value)
    for child in node.nodes:
        #print child.__dict__
        if child!=-1 and child.value == value or child.data =='T' or child.data =='F':
            # print  " child value is " + str(child.value)
            # print " get_child found value"
            return child
    # print "Error : get_child is returning None" + str(value)
    return None

def get_chi_value(split_map,num_one, num_zero, total):
    retVal = 0.0
    for k in split_map.keys():
        v = split_map[k]
        pos = num_one * len(v[1])/total
        neg = num_zero * len(v[1])/ total

        zero_count = v[1].count(0)
        ones_count = v[1].count(1)

        current = 0
        if ones_count != 0:
            current += pow(ones_count-pos, 2)/ones_count
        if zero_count != 0:
            current += pow(zero_count-neg, 2)/zero_count

        retVal += current
    return 1 - chi2.cdf(retVal, len(split_map.keys()))

#trainfeat, trainlab, best_index, unique_values
def get_split_map(train_data, train_label, index, values):
    retVal = dict()
    for value in values:
        split_data = [row for row in train_data if row[index] == value]
        split_label = [train_label[t] for t in range(len(train_label)) if train_data[t][index] == value]
        retVal[value] = [split_data, split_label]
    return retVal

def entropy(column):
    hashmap = dict()

    for val in column:
        hashmap[val] = hashmap.get(val, 0) + 1

    freqs = np.array(hashmap.values())
    # print "Freqs : " + str(freqs)
    # print len(column)
    freqs = freqs / len(column)
    retVal = 0.0
    for freq in freqs:
        if freq != 0:
            retVal += -(freq * np.log(freq))
            # print "Log : " + str(np.log(freq))
            # print "Freq: " + str(freq)
    return retVal


def get_unique_values(data, column):
    return {row[column] for row in data}

'''
TreeNode represents a node in your decision tree
TreeNode can be:
    - A non-leaf node: 
        - data: contains the feature number this node is using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take
        
    - A leaf node:
        - data: 'T' or 'F' 
        - children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.

'''

# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T', children = [-1]*5):
        self.nodes = list(children)
        self.data = data
        #print self.children

    def buildTree(self, trainfeat, trainlab, features_used, pval):
        #print "Hitting build tree"
        num_zero = trainlab.count(0)
        num_one = trainlab.count(1)
        if features_used.values().count('True') == num_feats:
            # we have used all features
            # print "All features used. Terminating"

            self.data = 'T' if num_one > num_zero else 'F'
            return
        elif num_one == 0 or num_zero == 0:
            # print "All zero or one. Terminating"
            self.data = 'F' if num_one == 0 else 'T'
            return
        else:
            # print "not terminating"
            best_index = self.pickBest(trainfeat, trainlab, features_used)
            # print "Best Index " + str(best_index)
            self.data = best_index
            dbg = "best_index was %s" % str(best_index)
            fp.write(dbg + "\n")
            # print "set self.data to best_index"
            unique_values = get_unique_values(trainfeat, best_index)
            #print "unique_values : " + str(unique_values)
            split_map = get_split_map(trainfeat, trainlab, best_index, unique_values)
            #print "split_map is : " + str(split_map.keys())
            chi_value = get_chi_value(split_map, num_one, num_zero, len(trainfeat))
            dbg = "chi_square was %s" % str(chi_value)
            fp.write(dbg + "\n")
            #print "Chi Value " + str(chi_value)
            if chi_value < pval:
                for k in split_map.keys():
                    #print "count, k is "+ str(count) +","+ str(k)
                    value = split_map[k]
                    new_features_used = copy.deepcopy(features_used)
                    new_features_used[best_index] = True
                    #print "Depth, K : " + str(depth) + ", " + str(k)
                    new_child = TreeNode()
                    new_child.buildTree(value[0],value[1], new_features_used, pval)
                    self.nodes[k-1] = new_child
                for i in range(5):
                    if (self.nodes[i] == -1):
                        self.nodes[i] = TreeNode()
            else:
                self.data = 'T' if num_one > num_zero else 'F'


    def pickBest(self, trainfeat, trainlab, features_used):
        label_entropy = entropy(trainlab)
        #print "Label Entropy " + str(label_entropy)
        max_gain = -1
        best_index = None

        for i in range(len(trainfeat[0])):
            if not features_used[i]:
                freq_map = get_column_data_map(trainfeat, i)
                current_entropy = 0.0
                for num in freq_map.keys():
                    current_label = [trainlab[t] for t in range(len(trainfeat)) if trainfeat[t][i] == num]
                    current_entropy += entropy(current_label)*(freq_map[num]/len(trainfeat))
                current_gain = label_entropy - current_entropy
                if current_gain > max_gain:
                    max_gain = current_gain
                    best_index = i
        return best_index


    def save_tree(self, filename):
        obj = open(filename, 'w')
        pkl.dump(self, obj)

def predict(root, input):

    current = root
    while current:
        if current.data == 'T': return '1'
        if current.data == 'F': return '0'
        index = int(root.data)-1
        current = current.nodes[input[index]-1]

    #return evaluate_datapoint(root.nodes[datapoint[int(root.data) - 1] - 1], datapoint)
    #    current = root
    #    #print current.data

    #    while current.data !='T' and current.data !='F':

    #        split_index = current.data
    #        value = input[split_index]
            #print value
            # print " trying to find child at level " + str(current_level)
    #        current = current.nodes[value-1]
            # if current == -1:return '0'
    #    return '1' if current.data == 'T' else '0'

from Queue import Queue
def treeIterator(node):
    curr = node
    q = Queue()
    q.put((node,0))
    while q.qsize()>0:
        for _ in range(q.qsize()):
            temp, level = q.get()
            # print "level : value" + str(level) + " : " + str(temp.value)
            if temp.data !='T' and temp.data !='F':
                for t in temp.nodes:
                    if t!= -1 :
                        q.put((t, level +1))
        # print ("\n")
        #else:
        #    break




# loads Train and Test data
def load_data(ftrain, ftest):
    Xtrain, Ytrain, Xtest = [], [], []
    with open(ftrain, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = map(int, row[0].split())
            Xtrain.append(rw)

    with open(ftest, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = map(int, row[0].split())
            Xtest.append(rw)

    ftrain_label = ftrain.split('.')[0] + '_label.csv'
    with open(ftrain_label, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = int(row[0])
            Ytrain.append(rw)

    # print('Data Loading: done')
    return Xtrain, Ytrain, Xtest


# A random tree construction for illustration, do not use this in your code!
def create_random_tree(depth):
    if (depth >= 7):
        if (random.randint(0, 1) == 0):
            return TreeNode('T', [])
        else:
            return TreeNode('F', [])

    feat = random.randint(0, 273)
    root = TreeNode(data=str(feat))

    for i in range(5):
        root.nodes[i] = create_random_tree(depth + 1)

    return root


if __name__ == "__main__":
    num_feats = 274
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', required=True)
    parser.add_argument('-f1', help='training file in csv format', required=True)
    parser.add_argument('-f2', help='test file in csv format', required=True)
    parser.add_argument('-o', help='output labels for the test dataset', required=True)
    parser.add_argument('-t', help='output tree filename', required=True)

    args = vars(parser.parse_args())

    pval = args['p']
    pval = float(pval)
    Xtrain_name = args['f1']
    Ytrain_name = args['f1'].split('.')[0] + '_labels.csv'  # labels filename will be the same as training file name but with _label at the end

    Xtest_name = args['f2']
    Ytest_predict_name = args['o']
    tree_name = args['t']

    Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)

    features_used = collections.defaultdict(bool)

    print("Training...")
    root = TreeNode()
    #print id(root)
    root.buildTree(Xtrain, Ytrain, features_used, pval)
    # treeIterator(root)
    #print id(root)
    #print root.__dict__
    root.save_tree(tree_name)
    print("Testing...")
    Ypredict = []

    #root.predict(Xtest[2811])
    # generate random labels
    for i in range(0, len(Xtest)):
         Ypredict.append(predict(root,Xtest[i]))

    # print Ypredict
    with open(Ytest_predict_name, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(Ypredict)

    print("Output files generated")
