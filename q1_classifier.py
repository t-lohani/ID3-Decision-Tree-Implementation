from __future__ import division

import collections
import copy
import numpy as np
import argparse
import sys

class TreeNode():
    def __init__(self, train_features, train_labels, threshold, data='T', children=[]):
        self.threshold = threshold
        self.train_features = train_features
        self.train_labels = train_labels
        self.nodes = []
        self.data = data
        self.buildDecisionTree()

    def prepareChildData(self, data, featureData, bestIndex, value):
        tempData = []
        for count, d in enumerate(featureData):
            if d == value:
                tempData.append(data[count])
        for row in tempData:
            del row[bestIndex]
        return tempData

    def filterArray(self, arr, val):
        filtered = []
        for count, item in enumerate(arr):
            if item == val:
                filtered.append(self.train_labels[count])
        return filtered

    def entropy(self, index, data = None):
        hashmap = dict()
        if data is None:
            column = [row[index] for row in self.train_features]
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
            column = [row[index] for row in self.train_features ]
        else:
            column = data

        freq_map = collections.Counter(column)
        count = len(self.train_labels)

        retVal = 0.0

        for k in freq_map.keys():

            filtered = self.filterArray(column, k)
            ent = self.entropy(-1, data = filtered)
            retVal += freq_map[k] * ent
        labelEntropy = self.entropy(-1, data = self.train_labels)
        t =  labelEntropy - retVal/count
        return t

    def pickBestAttr(self):
        bestGain = float('-inf')
        bestFeature = None
        bestIdx = 0
        for count, feature in enumerate(self.feature_name):
            currentGain = self.infoGain(count, data = None)
            if currentGain > bestGain:
                bestGain = currentGain
                bestFeature = feature
                bestIdx = count
        return bestIdx, bestFeature

    def buildDecisionTree(self):
        bestIdx, bestFeature =  self.pickBestAttr()
        print bestIdx, bestFeature
        featureData = [row[bestIdx] for row in self.train_features]
        unique_values = set(featureData)
        childAttributes = copy.deepcopy(self.feature_name)
        del childAttributes[bestIdx]
        for val in unique_values:
            childData =  self.prepareChildData(self.train_features, featureData, bestIdx, val)
            self.nodes.append(TreeNode(childData, self.train_labels, childAttributes,self.depth +1))


if __name__ == '__main__':
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    # https://pymotw.com/2/argparse/
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', default=float)
    parser.add_argument('-f1', default=str)
    parser.add_argument('-f2', default=str)
    parser.add_argument('-o', default=str)
    parser.add_argument('-t', default=str)

    t = parser.parse_args(sys.argv[1:])
    train_data = t.f1
    train_label = t.f1.split(".csv")[0] + "_label" +".csv"
    test_data = t.f2
    test_label = t.f2.split(".csv")[0] + "_label" + ".csv"

    file = open(train_data)
    for line in file:
        number_strings = line.split(' ')  # Split the line on runs of whitespace
        numbers = [int(n) for n in number_strings]  # Convert to integers
        train_features.append(numbers)  # Add the "row" to your list.

    file = open(train_label)
    for line in file:
        train_labels.append(int(line))

    # file = open('featnames.csv')
    # for line in file:
    #     feature_name.append(line)

    file = open(test_data)
    for line in file:
        number_strings = line.split(' ')
        numbers = [int(n) for n in number_strings]
        test_features.append(numbers)

    file = open(test_label)
    for line in file:
        test_labels.append(int(line))

    #labelEntropy = TreeNode.entropy(None, -1,data=train_labels )
    root = TreeNode(train_features, train_labels, t.p)
