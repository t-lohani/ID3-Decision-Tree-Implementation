from __future__ import division
import scipy.stats as st
import numpy as np
import math
# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data




    def entropy(self, index):
        hashmap = dict()

        column = [row[index] for row in train_features]
        for val in column:
            hashmap[val] = hashmap.get(val, 0) + 1
        freqs = np.array(hashmap.values())
        freqs = freqs / len(column)
        retVal = 0.0
        for freq in freqs:
            if freq!=0:
                retVal += -(freq*np.log(freq))
        return retVal

    def buildDecisionTree(self):
        print ""

# def main:
#     pass

if __name__ == '__main__':
    train_features = []
    train_labels = []

    train_feat_file = open('trainfeat.csv')
    for line in train_feat_file:
        number_strings = line.split(' ')  # Split the line on runs of whitespace
        numbers = [int(n) for n in number_strings]  # Convert to integers
        train_features.append(numbers)  # Add the "row" to your list.
    # print(train_features[39999])

    train_label_file = open('trainlabs.csv')
    for line in train_label_file:
        train_labels.append(int(line))
    #print(train_labels)
    print entropy(220)
