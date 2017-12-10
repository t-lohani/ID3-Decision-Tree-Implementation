from __future__ import division
import scipy.stats as st
import numpy as np
import math
import collections
# DO NOT CHANGE THIS CLASS


def filterArray(arr, val):
    return [ t for t in arr if t != val]

class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data




    def entropy(self, index, data = None):
        hashmap = dict()
        if data is None:
            column = [row[index] for row in train_features]
        else:
            column = data

        for val in column:
            hashmap[val] = hashmap.get(val, 0) + 1
        freqs = np.array(hashmap.values())
        freqs = freqs / len(column)
        retVal = 0.0
        for freq in freqs:
            if freq!=0:
                retVal += -(freq*np.log(freq))
        return retVal

    def infoGain(self, index, data=None):
        if data is None:
            column = [row[index] for row in train_features ]
        else:
            column = data

        freq_map = collections.Counter(column)
        #print freq_map
        count = len(train_labels)

        retVal = 0.0

        for k in freq_map.keys():



            #### need to filter something here.
            # filtered
            ent = self.entropy(-1, data = filtered)
            print " entropy is " + str(ent)
            retVal += freq_map[k] * ent

        t =  labelEntropy - retVal/count
        #print "info gain is " + str(t)
        return t

    def pickBestAttr(self):
        bestGain = float('-inf')
        bestFeature = None
        bestIdx = 0
        #print feature_name
        for count, feature in enumerate(feature_name):
            currentGain = self.infoGain(count, data = None)
            #print currentGain
            if currentGain > bestGain:
                bestGain = currentGain
                bestFeature = feature
                bestIdx = count
        return bestIdx, bestFeature

    def buildDecisionTree(self):
        print self.pickBestAttr()

# def main:
#     pass

if __name__ == '__main__':
    train_features = []
    train_labels = []
    feature_name = []

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

    feature_name_file = open('featnames.csv')
    for line in feature_name_file:
        feature_name.append(line)

    root = TreeNode()
    #print root.entropy(270)

    labelEntropy = root.entropy(-1, data=train_labels)
    #print train_features
    root.buildDecisionTree()