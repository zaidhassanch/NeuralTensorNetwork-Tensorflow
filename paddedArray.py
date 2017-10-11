import numpy as np 
import DnnData
import csv
import math
import tensorflow as tf
from scipy.io import loadmat
import os
import psutil

embedding_size = 100;
slice_size = 3;


def memoryUsage():
	pid = os.getpid()
	py = psutil.Process(pid)
	memoryUse = py.memory_info()[0]/1e6;  # memory use in GB...I think
	print('==> memory use:', memoryUse)





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
    tree = list(rows);
    print(tree[0])



lens = np.array([len(i) for i in tree])
print lens.shape
mask = np.arange(lens.max()) < lens[:,None]
out = np.zeros(mask.shape, dtype= np.int)
out[mask] = np.concatenate(tree)
#print out

E_matrix = np.zeros(shape = (100, 67448))
matVars = loadmat(dataPath + 'wordEmbed.mat');
word_embeds = matVars['E'];

#rint word_embeds[:, [0]]

#print word_embeds.shape
E_matrix[:,1:] = word_embeds
#print E_matrix

print memoryUsage()



tree_holder       = tf.placeholder(tf.int32,   [38696,None])
treeLength_holder = tf.placeholder(tf.float32, [38696,])
E_holder          = tf.placeholder(tf.float32, [100,67448])

W1_shape = [embedding_size, embedding_size, slice_size]
W1 = tf.Variable(tf.ones(shape=W1_shape))# * 2 * r - r;
#W2 = np.ones(shape = (embedding_size * 2, slice_size,data.num_relations))# * 2 * r - r;
#b1 = np.ones(shape = (1, slice_size, data.num_relations));
#U = np.ones(shape = (slice_size, 1, data.num_relations));


treeLengths = tf.reshape(treeLength_holder, [38696,1])

Emat = tf.transpose(E_holder);
collectedVectors = tf.gather(Emat, tree_holder)
sumVecs = tf.reduce_sum(collectedVectors, axis = 1)
entVec = tf.divide(sumVecs,treeLengths)

print 'first loop', memoryUsage();



with tf.Session() as session:
	print 'before session', memoryUsage()
	entVecRet = session.run(entVec, feed_dict={tree_holder: out,treeLength_holder: lens, E_holder: E_matrix})
	#print r;
	#print result
	#print result.shape;
	print entVecRet.shape
	print entVecRet[38693]
