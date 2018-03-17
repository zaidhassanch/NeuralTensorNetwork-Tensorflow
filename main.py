from sklearn import metrics
import numpy as np
import numpy as np  
import math
import tensorflow as tf
from scipy.io import loadmat
import os
import psutil
import ntnEval
import scipy.io
import pickle
from ntn import NTN
import DnnData

def makeSummary(data, writer, sess, merged, indexes = ""):
    feeddict = ntnNetwork.makeFeedDict(data, indexes);
    summary = sess.run(merged, feed_dict=feeddict)
    writer.add_summary(summary, i)
    writer.flush()

    

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


dataSet  = 'Wordnet/';
dataPath = '../data/' + dataSet;
savePath = '../output/';

# Can be changed latter
with open('DnnData_data.pkl', 'rb') as inputFile:
    data = pickle.load(inputFile)
    testData = pickle.load(inputFile)
    devData = pickle.load(inputFile)

dataRows = len(data.e1)
out, lens, E_matrix = DnnData.loadVars(dataPath);

DnnData.DnnData.out  = out;
DnnData.DnnData.lens = lens;

print E_matrix[1:4,1:4]



ntnNetwork          = NTN(E_matrix, data);
merged, e1,scorePosNet, loss, train_op = ntnNetwork.buildGraph();
init = tf.global_variables_initializer();

with tf.Session() as session:
	train_writer, test_writer, saver = ntnNetwork.saveOps(savePath,session);
	session.run(init);
	
	
	for i in xrange(200):
		print 'iter:', i;
		batches = dataRows // batch_size;
		
		for j in xrange(1):

			ntnNetwork.train(session, data);
			
			

		best_threshold = ntnEval.bestThreshold(devData, ntnNetwork, session, scorePosNet);
		print best_threshold;
		

		testAccuracy = ntnEval.findAccuracy(testData, data, ntnNetwork, session, scorePosNet, best_threshold);

		print "Flow completed", testAccuracy;
		exit();

		if(i%5 == 0):
			makeSummary(data, train_writer, session, merged, indexes);



