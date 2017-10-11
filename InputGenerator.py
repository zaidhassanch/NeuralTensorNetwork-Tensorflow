import numpy as np 
import DnnData
from sklearn.preprocessing import normalize
from scipy import spatial
from nueralTensorCost import tensorCost
import pandas as pd
import csv
import math
flipType = 0;

print "Starting DNN Network ..."
batch_size = 1000;
embedding_size = 100;
slice_size = 3;


# tree ids is going to be used
# initEmbed's We is going to be used
# how are the words going to get used
# how are the text files going to be read


dataSet  = 'Wordnet/'
dataPath = '../data/' + dataSet;
savePath = '../output/'

mUnitVec = np.zeros((100), dtype=np.float32)
mUnitVec[0] = 1.0;

with open(dataPath + 'tree_ids.csv') as csvfile:	#ids will need to have 1 subtracted off them
    rows = csv.reader(csvfile)
    tree = list(rows)
    print(tree[0])

data = DnnData.dataGen(dataPath, 'entities.txt', 'train.txt', 'relations.txt', 1);

print data.e1[0:10];
print data.relations[0:100];
print sum(data.relations[0:100] == 0);
print data.e2[0:10];

print data.e1.dtype;
print 'size of file', len(data.relations)
totalRelations = len(data.relations);
data.e3 =  np.zeros(shape=(totalRelations * 1, 1), dtype=np.int) # equivalent to matlab ones



	
# initialization
#W1{i} = rand(params.embedding_size, params.embedding_size, params.slice_size) * 2 * r - r;

W1 = np.ones(shape = (embedding_size, embedding_size, slice_size, data.num_relations))# * 2 * r - r;
W2 = np.ones(shape = (embedding_size * 2, slice_size,data.num_relations))# * 2 * r - r;
b1 = np.ones(shape = (1, slice_size, data.num_relations));
U = np.ones(shape = (slice_size, 1, data.num_relations));




print 'size of file', len(data.e1)

cost = tensorCost(data, W1, W2, b1, U, tree, flipType);

print cost;
exit();
"""





    #train_writer, test_writer, saver = dnnNet.saveOps(savePath,sess);
    
    #saver.restore(sess, "./Freebase_checkpoints/hidden=1000/model.ckpt")
    #batches = train_size // batch_size
    #epoch_loss = []
    #prev_acc = 0
    #print batches

"""