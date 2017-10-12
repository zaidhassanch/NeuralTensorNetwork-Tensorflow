import pickle
import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle
from scipy import spatial
from scipy.io import loadmat


relat_length = 50
bias = 0.5
relation_no = 13



class DnnData():
    def __init__(self, e1, e2, relat, yRef = "", entity_list = "", entity_length = "", num_relations = ""):
        self.e1           = e1;            # index of (vector or english world)
        self.e2           = e2;
        self.relations    = relat;
        self.yRef         = yRef;
        self.batchSize    = 0;
        self.num_relations = num_relations;
        self.entity_list  = entity_list;
        self.length       = len(self.e1)
        self.entity_length  = entity_length;

    def shuffle(self):
        self.e1, self.e2, self.relations, self.yRef = shuffle(self.e1, self.e2, self.relations, self.yRef);

    def load(self):
        self.batchSize = -1;

    def getFirst(self, size):
        index    = size;
        tempData = DnnData(self.e1[:index], self.e2[:index], self.relations[:index], self.yRef[:index], self.entity_list)
        return tempData;

    def getLast(self, percent):
        index    = int(self.e1.size * percent);
        tempData = DnnData(self.e1[index:], self.e2[index:], self.relations[index:], self.yRef[index:], self.entity_list)
        return tempData;

    def append(self, otherData):
        print((self.e1.shape))
        print((otherData.e1.shape))

        e1 = np.concatenate((self.e1, otherData.e1), axis=0);
        e2 = np.concatenate((self.e2, otherData.e2), axis=0);
        relations = np.concatenate((self.relations, otherData.relations), axis=0);
        yRef = np.concatenate((self.yRef, otherData.yRef), axis=0);
        return DnnData(e1, e2, relations,yRef, self.entity_list)


    def size(self):
        return self.e1.size;

    def printEnglish(self, i1, i2):
        print("-----")
        print(self.e1[i1:i2])
        for i in range(i1, i2):
            print i, self.entity_list[self.e1[i]], self.relations[i], self.entity_list[self.e2[i]] 

    def printx(self):
        print "========================"
        print 'e1',self.e1.shape
        print 'e2', self.e2.shape
        print 'relations',self.relations.shape
        print 'yRef', self.yRef.shape
        print "========================"
        print self.e1.shape
        print self.e1[0:5]

    def getRelationTriplet(self, relationString):
        relation = relation_dict[relationString];
        relation_loc = np.where(self.relations == relation)
        print relation_loc
        e1_new   = self.e1[relation_loc]
        e2_new   = self.e2[relation_loc]
        print("relations ==");
        relats   = self.relations[relation_loc]
        print relats
        yRef_new = self.yRef[relation_loc]
        data = DnnData(e1_new, e2_new, relats, yRef_new, self.entity_list);
        return data

    



def dataGen(path, entityFile, dataFile, relationFile):
    entityFile   = path + entityFile;
    dataFile      = path + dataFile;
    relationFile      = path + relationFile;

    with open(entityFile) as f:
        entity_list = f.readlines();
        entity_list = list([entity.strip('\n') for entity in entity_list]);
        entity_length = len(entity_list);
        print(len(entity_list));
    with open(relationFile) as f:
        relation_list = f.readlines();
        relation_list = list([relation.strip('\n') for relation in relation_list]);
        relation_length = len(relation_list);
        print(len(relation_list));


    with open(dataFile) as f:
        train_rows = f.readlines();
        train_rows = list([row.strip('\n').split() for row in train_rows]);
        print(len(train_rows));

    print(train_rows[0]);
    e1 = [];
    for row in train_rows:
        e1.append(entity_list.index(row[0]));
    relationsFinal = [];
    for row in train_rows:
        relationsFinal.append(relation_list.index(row[1]));
    e2 = [];
    for row in train_rows:
        e2.append(entity_list.index(row[2]));

    e1 = np.asarray(e1);
    e2 = np.asarray(e2);
    relationsFinal = np.asarray(relationsFinal);

    matVars = loadmat(path + 'wordEmbed.mat');          # same name everywhere
    word_embeds = matVars['E'];
    
    print word_embeds[0];
    data = DnnData(e1, e2, relationsFinal, entity_list = entity_list, entity_length = entity_length, num_relations = relation_length);
    data.word_embeds = word_embeds;
    return data;

def minMaxDist(x,vec): 
    distances = []
    for r in x:
        cosineDistance = spatial.distance.cosine(r, vec)
        distances.append(cosineDistance)
    distances = np.asarray(distances)
    print 'distance shape',distances.shape
    maxDist = np.amax(distances)
    minDist = np.amin(distances) 

    return minDist, maxDist

def meanCosDist(x,vec):
    distances = []
    
    for r in x:
        cosineDistance = spatial.distance.cosine(r, vec)
        distances.append(cosineDistance)
    distances = np.asarray(distances)
    return np.mean(distances)
