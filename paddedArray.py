import numpy as np 
import DnnData
import csv
import math
import tensorflow as tf
from scipy.io import loadmat
import os
import psutil
import DnnData
import random
import scipy.io

embedding_size = 100;
slice_size = 3;
no_of_entities = 38696;
flipType = 0;
batch_size = 20000;


def memoryUsage():
	pid = os.getpid()
	py = psutil.Process(pid)
	memoryUse = py.memory_info()[0]/1e6;  # memory use in GB...I think
	print('==> memory use:', memoryUse)


print "Starting DNN Network ..."

embedding_size = 100;
slice_size   = 3;
corrupt_size = 10;


# tree ids is going to be used
# initEmbed's We is going to be used
# how are the words going to get used
# how are the text files going to be read


dataSet  = 'Wordnet/'
dataPath = '../data/' + dataSet;
savePath = '../output/'
initialPath = '../data/' + dataSet + 'initialize.mat';
lstE3Path = '../data/' + dataSet + 'lstE3.mat';
mat = scipy.io.loadmat(initialPath)
mat2 = scipy.io.loadmat(lstE3Path)
W1Mat = mat['W1Mat'];
W2Mat = mat['W2Mat'];
e3Mat = np.squeeze(mat2['e3']) - 1;
lstMat = np.squeeze(mat2['lst']) - 1;



data = DnnData.dataGen(dataPath, 'entities.txt', 'train.txt', 'relations.txt');
dataRows = len(data.e1)

testData = DnnData.dataGen(dataPath, 'entities.txt', 'test.txt', 'relations.txt');
testRows = len(testData.e1)

devData = DnnData.dataGen(dataPath, 'entities.txt', 'dev.txt', 'relations.txt');
devRows = len(devData.e1)

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
matVars = loadmat(dataPath + 'initEmbed.mat');
word_embeds = matVars['We'];

print 'square ', np.sum(np.square(word_embeds))

#rint word_embeds[:, [0]]

#print word_embeds.shape
E_matrix[:,1:] = word_embeds
print 'square ', np.sum(np.square(E_matrix))
print E_matrix.dtype
#print E_matrix

print memoryUsage()


def update_x_2(inputVar):
	return inputVar




tree_holder       = tf.placeholder(tf.int32,   [no_of_entities,None]);
treeLength_holder = tf.placeholder(tf.float64, [no_of_entities,]);
#E_holder          = tf.placeholder(tf.float64, [embedding_size,67448]);		# use initial value stuff
e1_holder         = tf.placeholder(tf.int32,   [None,]);
e2_holder         = tf.placeholder(tf.int32,   [None,]);
relation_holder   = tf.placeholder(tf.int32,   [None,]);
e3_holder         = tf.placeholder(tf.int32,   [None,]);	# change above too
pred = tf.placeholder(tf.bool, shape=[])

E_Var = tf.Variable(dtype=tf.float64, initial_value=E_matrix,trainable=True)

#W1_shape = [embedding_size, embedding_size, slice_size, data.num_relations]; # change num_relations pos
#W1 = tf.Variable(tf.ones(shape=W1_shape, dtype = tf.float64)); #trun
#W1 = tf.Variable(tf.truncated_normal(shape=W1_shape, dtype = tf.float64, stddev = 6.0 / embedding_size)); #trun
W1 = tf.Variable(dtype=tf.float64, initial_value= W1Mat,trainable=True)
#W2_shape = [data.num_relations, embedding_size * 2, slice_size]; 
#W2 = tf.Variable(tf.ones(shape=W2_shape, dtype = tf.float64));
#W2 = tf.Variable(tf.random_uniform(shape=W2_shape, dtype = tf.float64));	#randuni
W2 = tf.Variable(dtype=tf.float64, initial_value= W2Mat,trainable=True)
# b1 and u are extremely simple things
b1_shape = [data.num_relations, 1, slice_size,];
b1 = tf.Variable(tf.zeros(shape=b1_shape, dtype = tf.float64));
U_shape = [data.num_relations, 1, slice_size,];
U = tf.Variable(tf.ones(shape=U_shape, dtype = tf.float64));

cost      = tf.Variable(0, dtype = tf.float64)
scorePosNet = tf.Variable(tf.zeros(shape = [0,1], dtype = tf.float64));
batchSize = tf.constant(batch_size, dtype = tf.float64)
reg_param = tf.constant(0.0001, dtype = tf.float64)
#x2 = tf.Variable([5])
treeLengths = tf.reshape(treeLength_holder, [no_of_entities,1]);
Emat = tf.transpose(E_Var);
collectedVectors = tf.gather(Emat, tree_holder);
sumVecs = tf.reduce_sum(collectedVectors, axis = 1);
entVec = tf.divide(sumVecs,treeLengths);

# look to eliminate
for i in xrange(data.num_relations):		#

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

	scorePosNet = tf.concat([scorePosNet, score_pos], 0);

	bias = tf.constant(1, dtype = tf.float64);

	indx = tf.where(tf.greater(score_pos + bias, score_neg))
	indxJogar = tf.gather(tf.transpose(indx), 0);
	scorePosRel = tf.gather(score_pos, tf.transpose(indxJogar));
	scoreNegRel = tf.gather(score_neg, tf.transpose(indxJogar));

	cost = cost + tf.reduce_sum((scorePosRel + bias) - scoreNegRel);

squareSum = tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2)) + tf.reduce_sum(tf.square(b1));
squareSum = squareSum +  tf.reduce_sum(tf.square(E_Var)) + tf.reduce_sum(tf.square(U));
loss = tf.divide(cost,batchSize) + reg_param / 2.0 * squareSum ;
#train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
train_op = tf.train.MomentumOptimize( 1e-3, 0.9, use_nesterov=False).minimize(loss);

"""
train_step = tf.contrib.opt.ScipyOptimizerInterface(
                cost,
                method='L-BFGS-B',
                options={'maxiter': 5})
"""

init = tf.global_variables_initializer();


print 'first loop', memoryUsage();



with tf.Session() as session:
	print 'before session', memoryUsage()

	session.run(init);
	
	for i in xrange(100):
		print 'iter:', i;
		batches = dataRows // batch_size;
		for j in xrange(batches):
			#indexes = range(j*batch_size,(j+1)*batch_size)
			indexes = np.random.randint(0,dataRows,size = batch_size)
			#print indexes.shape
			relMake = np.ravel(np.matlib.repmat(data.relations[indexes], 1, corrupt_size))
			e1Make  = np.ravel(np.matlib.repmat(data.e1[indexes], 1, corrupt_size))
			e2Make  = np.ravel(np.matlib.repmat(data.e2[indexes], 1, corrupt_size))
			e3Make  = np.random.randint(0, data.entity_length, size=(batch_size * corrupt_size))
			# this should not be starting from 1


			if (random.uniform(0, 1) > 0.5):
				flip 	= False;
			else:
				flip 	= False;
			
			#flip 	= True;
			
			lossRet, squareRet, _ = session.run([loss, squareSum, train_op], 
				feed_dict={tree_holder: out,
				treeLength_holder: lens, 
				e1_holder        : e1Make,
				e2_holder        : e2Make,
				relation_holder  : relMake,
				e3_holder        : e3Make,
				pred			 : flip}); 

			#train_step.minimize(session, feed_dict)
			#costRet = cost.eval(feed_dict = feed_dict);
			print lossRet;




		# just the accuracy reproduced please
		# just a dummy this 
		devData.e3  = np.zeros(shape=(devRows * corrupt_size), dtype=np.int)
		
		predictions, e1Ret = session.run([scorePosNet, e1], 
		feed_dict={tree_holder: out,
				treeLength_holder: lens, 
				e1_holder        : devData.e1,
				e2_holder        : devData.e2,
				relation_holder  : devData.relations,
				e3_holder        : devData.e3,
				pred			 : True})
		
		predictions = np.ravel(predictions) # Jogar step
		#print predictions
		#print e1Ret

		# max and min of predictions
		# find best here
		rmax = np.amax(predictions);
		lmax = np.amin(predictions);

		#print rmax

		best_threshold = np.ones(shape= (data.num_relations, 1)) * lmax;
		best_acc = np.ones(shape= (data.num_relations, 1)) * (-1);
		ySet = np.array([True, False], dtype=np.bool)  # put in the false
		yGroundAll = np.ravel(np.matlib.repmat(ySet, 1, devRows // 2))

		while lmax <= rmax:
			yRetPred = (predictions <= lmax);
			start = 0;
			for i in xrange(data.num_relations):
				lst = (devData.relations == i);
				yGnd = yGroundAll[lst];

				end = start + len(yGnd);

				accuracy = np.mean(yRetPred[start:end] == yGnd);
				start = end;

				if accuracy > best_acc[i]:
					best_acc[i]       = accuracy; 
					best_threshold[i] = lmax;

			lmax = lmax + 0.01;


			
		#print best_threshold;
		#print best_acc;	

		# just a dummy this
		testData.e3 = np.zeros(shape=(testRows * corrupt_size), dtype=np.int)

		predictions, e1Ret = session.run([scorePosNet, e1],
										 feed_dict={tree_holder: out,
													treeLength_holder: lens,
													e1_holder: testData.e1,
													e2_holder: testData.e2,
													relation_holder: testData.relations,
													e3_holder: testData.e3,
													pred			: True})

		predictions = np.ravel(predictions) # Jogar step
		ySet = np.array([True, False], dtype=np.bool)  # put in the false
		yGroundAll = np.ravel(np.matlib.repmat(ySet, 1, testRows // 2))

		testAccSum = 0.0;
		start = 0;
		for i in xrange(data.num_relations):
			lst = (testData.relations == i);
			yGnd = yGroundAll[lst];
			yRetPred = (predictions <= best_threshold[i]);
			end = start + len(yGnd);

			accuracySum = np.sum(yRetPred[start:end] == yGnd);
			testAccSum = testAccSum + accuracySum;
			start = end;

		print 'test accuracy: ', (testAccSum / testRows)
		#print r;
	#print result
	#print result.shape;
	#print entVecRet.shape

	#print entVecRet[38693]
	#print aRet
	#print concatRet

