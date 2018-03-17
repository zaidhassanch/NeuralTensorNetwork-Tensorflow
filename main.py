from sklearn import metrics
import numpy as np 
import csv
import math
import tensorflow as tf
from scipy.io import loadmat
import os
import psutil
import random
import scipy.io
import pickle
from ntn import NTN

no_of_entities = 38696;
flipType = 0;
batch_size = 20000;
slice_size = 3;
corrupt_size = 10;



print "Starting DNN Network ..."




# tree ids is going to be used
# initEmbed's We is going to be used
# how are the words going to get used
# how are the text files going to be read


dataSet  = 'Wordnet/'
dataPath = '../data/' + dataSet;
savePath = '../output/'
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


out, E_matrix

print E_matrix[1:4,1:4]
exit();


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
			_,lossRet = session.run([train_op, loss] , feeddict_new);	# first Neg is wrong
			print 'loss', lossRet;	


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
