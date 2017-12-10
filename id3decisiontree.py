from __future__ import division

import collections
import copy
import numpy as np
current_count = 0

# DO NOT CHANGE THIS CLASS


class TreeNode():
    def __init__(self, train_features, train_labels, feature_name,depth, data='T',children=[-1]*5):
        self.depth = depth
        #if self.depth >=5:
        #    return
        self.train_features = train_features
        self.train_labels = train_labels
        self.feature_name = feature_name
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
        #print freq_map
        count = len(self.train_labels)

        retVal = 0.0

        for k in freq_map.keys():

            filtered = self.filterArray(column, k)
            ent = self.entropy(-1, data = filtered)
            #print " entropy is " + str(ent)
            retVal += freq_map[k] * ent
        labelEntropy = self.entropy(-1, data = self.train_labels)
        t =  labelEntropy - retVal/count
        return t

    def pickBestAttr(self):
        bestGain = float('-inf')
        bestFeature = None
        bestIdx = 0
        #print self.feature_name
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
        # print len(childAttributes)
        for val in unique_values:
            childData =  self.prepareChildData(self.train_features, featureData, bestIdx, val)
            self.nodes.append(TreeNode(childData, self.train_labels, childAttributes,self.depth +1))


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
    #labelEntropy = TreeNode.entropy(None, -1,data=train_labels )
    root = TreeNode(train_features, train_labels, feature_name,0)