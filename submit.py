from __future__ import division

import argparse
import collections
import copy
import csv
import pickle as pkl

import numpy as np
from scipy.stats import chi2

# Function to return a key value pair of value and its frequency for a column
def get_column_data_map(data, column):
    return collections.Counter([row[column] for row in data])

# Function to calculate the Chi Value.
def get_chi_value(split_map, num_one, num_zero, total):
    retVal = 0.0
    # Interating over the map
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
    # Return the chi-value
    return 1 - chi2.cdf(retVal, len(split_map.keys()))

# Function to split the map based on index
def get_split_map(train_data, train_label, index, values):
    retVal = dict()
    for value in values:
        split_data = [row for row in train_data if row[index] == value]
        split_label = [train_label[t] for t in range(len(train_label)) if train_data[t][index] == value]
        retVal[value] = [split_data, split_label]
    return retVal

# Function to calculate entropy of one column
def entropy(column):
    hashmap = dict()

    for val in column:
        hashmap[val] = hashmap.get(val, 0) + 1

    freqs = np.array(hashmap.values())
    freqs = freqs / len(column)
    retVal = 0.0
    for freq in freqs:
        if freq != 0:
            retVal += -(freq * np.log(freq))
    return retVal

# Function to get one particular column from the data set
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
    # Constructor of Tree Node
    def __init__(self, data='T', children = [-1]*5):
        self.nodes = list(children)
        self.data = data

    # Function to build tree recursively
    def buildTree(self, trainfeat, trainlab, features_used, pval):
        global numberOfNodes;
        numberOfNodes += 1
        # Counting number of zeroes and ones.
        num_zero = trainlab.count(0)
        num_one = trainlab.count(1)

        # IF all the features have been used, stop splitting and assign the node value as leaf by marking it by 'T'
        # or 'F' based on the count of ones and zeroes
        if features_used.values().count('True') == num_feats:
            self.data = 'T' if num_one > num_zero else 'F'
            return
        # If the column has only 0 or 1 values, mark it as leaf node.
        # Assign 'T' if all the values are 1 and 'F' if they are 0.
        elif num_one == 0 or num_zero == 0:
            self.data = 'F' if num_one == 0 else 'T'
            return
        # Adding children to the node and recursively building tree.
        else:
            # Picking the best attribute index on which we should split at this node.
            best_index = self.pickBest(trainfeat, trainlab, features_used)
            self.data = best_index

            # Creating unique values and split map
            unique_values = get_unique_values(trainfeat, best_index)
            split_map = get_split_map(trainfeat, trainlab, best_index, unique_values)

            # Calculatng chi value for stopping criteria.
            chi_value = get_chi_value(split_map, num_one, num_zero, len(trainfeat))

            # If chi value is less than the passed threshold value, adding children recursively
            if chi_value < pval:
                # Iterating on the child map and adding child TreeNodes.
                for k in split_map.keys():
                    value = split_map[k]
                    new_features_used = copy.deepcopy(features_used)
                    new_features_used[best_index] = True
                    new_child = TreeNode()
                    # Building tree recursively
                    new_child.buildTree(value[0],value[1], new_features_used, pval)
                    # Adding child to the children array
                    self.nodes[k-1] = new_child

                # Adding dummy nodes with default value of 'F' for empty children
                for i in range(5):
                    if (self.nodes[i] == -1):
                        self.nodes[i] = TreeNode(data='F')
            # If the chi value is greater than the threshold passed, mark the node as leaf.
            else:
                self.data = 'T' if num_one > num_zero else 'F'

    # Function to return the best index to split on using minimum entopy
    def pickBest(self, trainfeat, trainlab, features_used):
        # Calculating label entopy based on train label
        label_entropy = entropy(trainlab)
        max_gain = -1
        best_index = None


        for i in range(len(trainfeat[0])):
            if not features_used[i]:
                # Calculating entopy
                freq_map = get_column_data_map(trainfeat, i)
                current_entropy = 0.0
                for num in freq_map.keys():
                    current_label = [trainlab[t] for t in range(len(trainfeat)) if trainfeat[t][i] == num]
                    current_entropy += entropy(current_label)*(freq_map[num]/len(trainfeat))

                # Calculating gain
                current_gain = label_entropy - current_entropy
                if current_gain > max_gain:
                    max_gain = current_gain
                    best_index = i

        return best_index


    def save_tree(self, filename):
        obj = open(filename, 'w')
        pkl.dump(self, obj)

# Function to traverse the tree and predict the output on a given input
def predict(root, input):
    if root.data == 'T': return 1
    if root.data =='F': return 0
    return predict(root.nodes[input[int(root.data)-1]-1], input)

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

    return Xtrain, Ytrain, Xtest

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
    Ytrain_name = args['f1'].split('.')[0] + '_label.csv'

    Xtest_name = args['f2']
    Ytest_predict_name = args['o']
    tree_name = args['t']

    Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)

    features_used = collections.defaultdict(bool)

    numberOfNodes = 0

    print("Training...")
    root = TreeNode()
    root.buildTree(Xtrain, Ytrain, features_used, pval)
    root.save_tree(tree_name)
    print("Number of nodes created " + str(numberOfNodes))
    print("Testing...")
    Ypredict = []

    for i in range(0, len(Xtest)):
         Ypredict.append(predict(root,Xtest[i]))

    with open(Ytest_predict_name, "wb") as f:
        for item in Ypredict:
            f.write("%s\n"%item)

    print("Output files generated")