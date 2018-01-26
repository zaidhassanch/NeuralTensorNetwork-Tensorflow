from sklearn import metrics
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
import pickle
from ntn import NTN

no_of_entities = 38696;
flipType = 0;
batch_size = 20000;
slice_size = 3;
corrupt_size = 10;


def nanargmax(a):
    idx = np.argmax(a, axis=None)
    multi_idx = np.unravel_index(idx, a.shape)
    if np.isnan(a[multi_idx]):
        nan_count = np.sum(np.isnan(a))
        # In numpy < 1.8 use idx = np.argsort(a, axis=None)[-nan_count-1]
        idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
        multi_idx = np.unravel_index(idx, a.shape)
    return multi_idx

def memoryUsage():
	pid = os.getpid()
	py = psutil.Process(pid)
	memoryUse = py.memory_info()[0]/1e6;  # memory use in GB...I think
	print('==> memory use:', memoryUse)


print "Starting DNN Network ..."




# tree ids is going to be used
# initEmbed's We is going to be used
# how are the words going to get used
# how are the text files going to be read


dataSet  = 'Wordnet/'
dataPath = '../data/' + dataSet;
savePath = '../output/'
initialPath = '../data/' + dataSet + 'initialize.mat';
"""
valuesPath = '../data/' + dataSet + 'regValues.mat';
lstE3Path = '../data/' + dataSet + 'lstE3.mat';
forwardValsPath = '../data/' + dataSet + 'forwardValues.mat';
specificValsPath = '../data/' + dataSet + 'specificValues.mat';
outputValsPath = '../data/' + dataSet + 'outputValues_orig.mat';
"""
mat       = scipy.io.loadmat(initialPath);
"""
valuesMat = scipy.io.loadmat(valuesPath);
mat2 = scipy.io.loadmat(lstE3Path);
forwardMat = scipy.io.loadmat(forwardValsPath);
specificMat = scipy.io.loadmat(specificValsPath);
"""


W1Mat = mat['W1Mat'];
W2Mat = mat['W2Mat'];
"""
gradW1mat = valuesMat['gradW1mat'];
gradW2mat = valuesMat['gradW2mat'];
gradb1mat = valuesMat['gradb1mat'];
gradUmat = valuesMat['gradUmat'];
gradEmat = valuesMat['gradE'];
gradEntmat = valuesMat['entVecGrad'];
gradSpecMat = specificMat['entVecGrad_specific'];
"""

#e3Mat = np.squeeze(mat2['e3']) - 1;
#lstMat = np.squeeze(mat2['lst']) - 1;
#scorePosMat = forwardMat['score_pos'];
#scoreNegMat = forwardMat['score_neg'];

with open('DnnData_data.pkl', 'rb') as inputFile:
    data = pickle.load(inputFile)
    testData = pickle.load(inputFile)
    devData = pickle.load(inputFile)

dataRows = len(data.e1)
testRows = len(testData.e1)
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

E_matrix = np.zeros(shape = (100, 67448)); 	# As opposed to zeros to ensure error warning
matVars = loadmat(dataPath + 'initEmbed.mat');
word_embeds = matVars['We'];
print 'square ', np.sum(np.square(word_embeds))
E_matrix[:,1:] = word_embeds
print 'square ', np.sum(np.square(E_matrix))
print E_matrix.dtype
print memoryUsage()

ntnNetwork          = NTN(E_matrix, data);
gradsEntVec, gradsE,scorePosNet, e1, train_op = ntnNetwork.buildGraph();


init = tf.global_variables_initializer();

print 'first loop', memoryUsage();

with tf.Session() as session:
	print 'before session', memoryUsage()

	session.run(init);
	bestAccuracy = 0.0;
	
	for i in xrange(200):
		print 'iter:', i;
		batches = dataRows // batch_size;
		
		for j in xrange(batches):
			
			#indexes = range(j*batch_size,(j+1)*batch_size)
			indexes = np.random.randint(0,dataRows,size = batch_size)
			#print indexes.shape		
			#data.e3Make = e3Mat; Or inside
			# this should not be starting from 1


			if (random.uniform(0, 1) > 0.5):
				data.flip 	= True;
			else:
				data.flip 	= False;
			data.lens = lens;
			data.out  = out;
			
			#flip 	= True;
        
			feeddict_new = ntnNetwork.makeFeedDict(data, indexes, 10); # indexes or lstMat
			geVec, gE, _ = session.run([gradsEntVec, gradsE, train_op] , feeddict_new);	# first Neg is wrong


		# just the accuracy reproduced please
		# just a dummy this 
		devData.e3Make  = np.zeros(shape=(devRows * corrupt_size), dtype=np.int)
		devData.out = out;
		devData.lens = lens;
		devData.flip = True;
		feeddict_new = ntnNetwork.makeFeedDict(devData);

		predictions, e1Ret = session.run([scorePosNet, e1], feeddict_new)
		
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

		# just a dummy this
		testData.e3Make = np.zeros(shape=(testRows * corrupt_size), dtype=np.int)
		testData.out = out;
		testData.lens = lens;
		testData.flip = True;

		for j in xrange(2):
			if(j == 0):
				feeddict_new = ntnNetwork.makeFeedDict(testData);
			elif(j == 1):
				feeddict_new = ntnNetwork.makeFeedDict(data);
			

			predictions, e1Ret = session.run([scorePosNet, e1],feeddict_new);

			predictions = np.ravel(predictions) # Jogar step
			ySet = np.array([True, False], dtype=np.bool)  # put in the false

			if(j == 0):
				yGroundAll = np.ravel(np.matlib.repmat(ySet, 1, testRows // 2));
			elif(j == 1):
				print dataRows
				yGroundAll = np.ravel(np.matlib.repmat(ySet, 1, dataRows // 2));
				print "yGround 112", yGroundAll.shape;

			

			testAccSum = 0.0;
			start = 0;
			for i in xrange(data.num_relations):
				if(j == 0):
					lst = (testData.relations == i);
				elif(j == 1):
					lst = (data.relations == i);
				
				print yGroundAll.shape;
				print lst.shape;
				yGnd = yGroundAll[lst];
				


				yRetPred = (predictions <= best_threshold[i]);
				end = start + len(yGnd);

				accuracySum = np.sum(yRetPred[start:end] == yGnd);
				testAccSum = testAccSum + accuracySum;
				start = end;
			if(j == 0):
				print 'test accuracy: ', (testAccSum / testRows);
			elif(j == 1):
				print 'train accuracy: ', (testAccSum / dataRows);		
		
