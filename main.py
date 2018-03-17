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




batch_size = 20000;

print "Starting DNN Network ..."


# tree ids is going to be used
# initEmbed's We is going to be used
# how are the words going to get used
# how are the text files going to be read


dataSet  = 'Wordnet/';
dataPath = '../data/' + dataSet;


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
ntnNetwork.createSession();

	
	
for i in xrange(200):
	print 'iter:', i;
	batches = dataRows // batch_size;
	
	for j in xrange(1):

		ntnNetwork.train(data);

	exit();
		
	testAccuracy = ntn.test(session, devData, data);

	print "Flow completed", testAccuracy;
	exit();

	if(i%5 == 0):
		makeSummary(data, train_writer, session, merged, indexes);



