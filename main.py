from sklearn import metrics
import numpy as np 
import math
import tensorflow as tf
from scipy.io import loadmat
import os
import psutil

import scipy.io
import pickle
from ntn import NTN
import DnnData

no_of_entities = 38696;
flipType = 0;
batch_size = 20000;
slice_size = 3;
corrupt_size = 10;

def bestThreshold(devData, session, scorePosNet):
	bestAccuracy = 0.0;

	# just the accuracy reproduced please
	# just a dummy this 
	devData.e3Make  = np.zeros(shape=(devRows * corrupt_size), dtype=np.int)
	devData.out = out;
	devData.lens = lens;
	devData.flip = True;
	feeddict_new = ntnNetwork.makeFeedDict(devData);

	predictions = session.run(scorePosNet, feeddict_new)
	
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
	return best_threshold;



print "Starting DNN Network ..."




# tree ids is going to be used
# initEmbed's We is going to be used
# how are the words going to get used
# how are the text files going to be read


dataSet  = 'Wordnet/';
dataPath = '../data/' + dataSet;
savePath = '../output/';

# no worries
initialPath = '../data/' + dataSet + 'initialize.mat';
mat       = scipy.io.loadmat(initialPath);
W1Mat = mat['W1Mat'];
W2Mat = mat['W2Mat'];


# Can be changed latter
with open('DnnData_data.pkl', 'rb') as inputFile:
    data = pickle.load(inputFile)
    testData = pickle.load(inputFile)
    devData = pickle.load(inputFile)

dataRows = len(data.e1)
testRows = len(testData.e1)
devRows = len(devData.e1)


out, lens, E_matrix = DnnData.loadVars(dataPath);
data.lens = lens;
data.out  = out;

print E_matrix[1:4,1:4]



ntnNetwork          = NTN(E_matrix, data);
merged, e1,scorePosNet, loss, train_op = ntnNetwork.buildGraph();
init = tf.global_variables_initializer();

def makeSummary(data, writer, sess, merged, indexes = ""):
    feeddict = ntnNetwork.makeFeedDict(data, indexes);
    summary = sess.run(merged, feed_dict=feeddict)
    writer.add_summary(summary, i)
    writer.flush()


with tf.Session() as session:
	train_writer, test_writer, saver = ntnNetwork.saveOps(savePath,session);
	session.run(init);
	
	
	for i in xrange(200):
		print 'iter:', i;
		batches = dataRows // batch_size;
		
		for j in xrange(1):

			ntnNetwork.train(session, data);
			
			

		best_threshold = bestThreshold(devData, session, scorePosNet);
		print best_threshold;
		exit();



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
			

			if(j == 0):
				ySet = np.array([True, False], dtype=np.bool)  # put in the false
				yGroundAll = np.ravel(np.matlib.repmat(ySet, 1, testRows // 2));
			elif(j == 1):
				ySet = np.array([True], dtype=np.bool)  # put in the false
				yGroundAll = np.ravel(np.matlib.repmat(ySet, 1, dataRows));
				print "yGround 112", yGroundAll.shape;

			

			testAccSum = 0.0;
			start = 0;
			for i in xrange(data.num_relations):
				if(j == 0):
					lst = (testData.relations == i);
				elif(j == 1):
					lst = (data.relations == i);
				
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
		if(i%5 == 0):
			makeSummary(data, train_writer, session, merged, indexes);



