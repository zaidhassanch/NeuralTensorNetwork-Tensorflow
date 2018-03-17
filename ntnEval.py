import tensorflow as tf
from ntn import NTN
import DnnData
import numpy as np

# to be removed soon
corrupt_size = 10;


def bestThreshold(devData, ntnNetwork, session, scorePosNet):
	bestAccuracy = 0.0;
	devRows = len(devData.e1)

	# just the accuracy reproduced please
	# just a dummy this 
	devData.e3Make  = np.zeros(shape=(devRows * corrupt_size), dtype=np.int)
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

	best_threshold = np.ones(shape= (devData.num_relations, 1)) * lmax;
	best_acc = np.ones(shape= (devData.num_relations, 1)) * (-1);
	ySet = np.array([True, False], dtype=np.bool)  # put in the false
	yGroundAll = np.ravel(np.matlib.repmat(ySet, 1, devRows // 2))

	while lmax <= rmax:
		yRetPred = (predictions <= lmax);
		start = 0;
		for i in xrange(devData.num_relations):
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


def findAccuracy(testData, data, ntnNetwork, session, scorePosNet, best_threshold):
			# just a dummy this
	testRows = len(testData.e1)
	dataRows = len(data.e1)

	testData.e3Make = np.zeros(shape=(testRows * corrupt_size), dtype=np.int)
	testData.flip = True;

	for j in xrange(2):
		if(j == 0):
			feeddict_new = ntnNetwork.makeFeedDict(testData);
		elif(j == 1):
			feeddict_new = ntnNetwork.makeFeedDict(data);
		

		predictions = session.run(scorePosNet,feeddict_new);

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
			testAccuracy = testAccSum / testRows
		elif(j == 1):
			print 'train accuracy: ', (testAccSum / dataRows);	

	return testAccuracy;

