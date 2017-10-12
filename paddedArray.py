import numpy as np 
import DnnData
import csv
import math
import tensorflow as tf
from scipy.io import loadmat
import os
import psutil
import DnnData

embedding_size = 100;
slice_size = 3;
no_of_entities = 38696;
flipType = 0;


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

data = DnnData.dataGen(dataPath, 'entities.txt', 'train.txt', 'relations.txt');
dataRows = len(data.e1)
data.e3 =  np.zeros(shape=(dataRows * 1), dtype=np.int)

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


def update_x_2(inputVar):
	return inputVar




tree_holder       = tf.placeholder(tf.int32,   [no_of_entities,None]);
treeLength_holder = tf.placeholder(tf.float32, [no_of_entities,]);
E_holder          = tf.placeholder(tf.float32, [embedding_size,67448]);
e1_holder         = tf.placeholder(tf.int32,   [dataRows,]);
e2_holder         = tf.placeholder(tf.int32,   [dataRows,]);
relation_holder   = tf.placeholder(tf.int32,   [dataRows,]);
e3_holder         = tf.placeholder(tf.int32,   [dataRows,]);
pred = tf.placeholder(tf.bool, shape=[])

W1_shape = [embedding_size, embedding_size, slice_size, data.num_relations]; # change num_relations pos
W1 = tf.Variable(tf.ones(shape=W1_shape));
W2_shape = [data.num_relations, embedding_size * 2, slice_size]; 
W2 = tf.Variable(tf.ones(shape=W2_shape));
b1_shape = [data.num_relations, 1, slice_size,];
b1 = tf.Variable(tf.ones(shape=b1_shape));
U_shape = [data.num_relations, 1, slice_size,];
U = tf.Variable(tf.ones(shape=U_shape));

cost  = tf.Variable(0, dtype = tf.float32)
#x2 = tf.Variable([5])
#y = tf.cond(pred, lambda: update_x_2(x), lambda: tf.identity(x))

#for i in xrange(data.num_relations):
#lst = (data.relations == i);
#U = np.ones(shape = (slice_size, 1, data.num_relations));



treeLengths = tf.reshape(treeLength_holder, [no_of_entities,1]);

Emat = tf.transpose(E_holder);
collectedVectors = tf.gather(Emat, tree_holder);
sumVecs = tf.reduce_sum(collectedVectors, axis = 1);
entVec = tf.divide(sumVecs,treeLengths);

# look to eliminate
for i in xrange(data.num_relations):

	lst = tf.where(tf.equal(relation_holder, i))
	e1 = tf.gather(e1_holder,lst);
	e2 = tf.gather(e2_holder,lst);
	e3 = tf.gather(e3_holder,lst);
	entVecE1 = tf.squeeze(tf.gather(entVec,e1));
	entVecE2 = tf.squeeze(tf.gather(entVec,e2));
	entVecE3 = tf.squeeze(tf.gather(entVec,e3));
	#entVecE1N = tf.Variable(tf.zeros(shape = entVecE1.get_shape()));

	entVecE1Neg = tf.cond(pred, lambda: update_x_2(entVecE1), lambda: update_x_2(entVecE3))
	entVecE2Neg = tf.cond(pred, lambda: update_x_2(entVecE3), lambda: update_x_2(entVecE2))
	e1Neg = tf.cond(pred, lambda: update_x_2(e1), lambda: update_x_2(e3))
	e2Neg = tf.cond(pred, lambda: update_x_2(e3), lambda: update_x_2(e2))

	W1transpose = tf.transpose(W1, perm=[3,2,0, 1]);
	# restrict yourself to special i
	W1specificTranspose = tf.gather(W1transpose,i);

	firstBi = tf.tensordot(W1specificTranspose, tf.transpose(entVecE2), axes = [[2], [0]]);
	firstBiNeg = tf.tensordot(W1specificTranspose, tf.transpose(entVecE2Neg), axes = [[2], [0]]);
	secondB = tf.multiply(tf.transpose(entVecE1), firstBi);
	secondBNeg = tf.multiply(tf.transpose(entVecE1Neg), firstBiNeg);
	finalBi = tf.reduce_sum(secondB , 1);
	finalBiNeg = tf.reduce_sum(secondBNeg , 1);

	W2specific = tf.gather(W2,i);
	b1specific = tf.gather(b1,i);
	concatEntVecs    = tf.concat([entVecE1, entVecE2], 1);
	concatEntNegVecs = tf.concat([entVecE1Neg, entVecE2Neg], 1);
	simpleProd    = tf.add(tf.matmul(concatEntVecs, W2specific), b1specific);
	simpleNegProd = tf.add(tf.matmul(concatEntNegVecs, W2specific), b1specific);

	v_pos = simpleProd    + tf.transpose(finalBi);
	v_neg = simpleNegProd + tf.transpose(finalBiNeg);
	z_pos = tf.tanh(v_pos);
	z_neg = tf.tanh(v_neg);

	UtransposeSpecific = tf.transpose(tf.gather(U,i));
	score_pos		   = tf.matmul(z_pos, UtransposeSpecific);
	score_neg		   = tf.matmul(z_neg, UtransposeSpecific);

	bias = tf.constant(1, dtype = tf.float32);

	indx = tf.where(tf.greater(score_pos + bias, score_neg))
	indxJogar = tf.gather(tf.transpose(indx), 0);
	scorePosRel = tf.gather(score_pos, tf.transpose(indxJogar));
	scoreNegRel = tf.gather(score_neg, tf.transpose(indxJogar));

	cost = cost + tf.reduce_sum((scorePosRel + bias) - scoreNegRel);





	#firstBiTranspose = tf.transpose(firstBi, perm = ????)

init = tf.global_variables_initializer();



	#v_pos[k, :] = sum(np.multiply(entVecE1,np.dot(W1[:, :, k, i], entVecE2)))
	# move second to last and simply dot

	# complete rest of the code with transposes


print 'first loop', memoryUsage();



with tf.Session() as session:
	print 'before session', memoryUsage()

	session.run(init);
	entVecRet, aRet, concatRet = session.run([score_neg,indx, cost], 
		feed_dict={tree_holder: out,
				treeLength_holder: lens, 
				E_holder         : E_matrix,
				e1_holder        : data.e1,
				e2_holder        : data.e2,
				relation_holder  : data.relations,
				e3_holder        : data.e3,
				pred			 : True})
	#print r;
	#print result
	#print result.shape;
	print entVecRet.shape
	print entVecRet
	#print entVecRet[38693]
	print aRet
	print concatRet

