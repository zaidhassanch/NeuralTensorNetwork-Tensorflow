import numpy as np 
import tensorflow as tf
import math
import time
import scipy.io
import warnings
import random
embedding_size = 100;
slice_size   = 3;
corrupt_size = 10;
slice_size = 3;
no_of_entities = 38696;
batch_size = 20000;
dataSet  = 'Wordnet/';
dataPath = '../data/' + dataSet;
initialPath = '../data/' + dataSet + 'initialize.mat';
mat       = scipy.io.loadmat(initialPath);

W1Mat = mat['W1Mat'];
W2Mat = mat['W2Mat'];



print W2Mat.shape;




class NTN():

    def __init__(self, entityEmbeds, data):
        self.E_matrix = entityEmbeds
        self.head_length   = 100   # this is a constant for this example
        self.num_relations = data.num_relations;
        self.loss         = "";
        self.train_op     = "";
        self.e1_holder    = "";
        self.e2_holder    = "";
        self.relat_holder = "";
        self.tree_holder  = "";
        self.learningRate = "";


    def makeHolderInt(self):
        holder = tf.placeholder(tf.int32, shape = (None,))
        return holder;

    def makeHolderInt2D(self):
        holder = tf.placeholder(tf.int32,   [no_of_entities,None]);
        return holder
    def makeHolderFloat(self):
        holder = tf.placeholder(tf.float64,   [no_of_entities,]);
        return holder
    def makeHolderBool(self):
        holder = tf.placeholder(tf.bool, shape=[]);
        return holder


    def makeLearningFloat(self):
        holder = tf.placeholder(tf.float64, [no_of_entities,])
        return holder

    def accuracy(self, y,y_pred):
        yret = (y_pred > 0.5)
        a = np.mean(y == yret)
        return a

    def makeFeedDict(self, data, indexes = "", corrupt_size = 1):

        data.e3Make  = np.random.randint(0, data.entity_length, size=(batch_size * corrupt_size));

        # Keep an eye on version of python for indexes == ""'s interpretation
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)

            if(indexes == ""):
                feeddict = {
                    self.e1_holder        :  np.ravel(np.matlib.repmat(data.e1, 1, corrupt_size)),
                    self.relation_holder  :  np.ravel(np.matlib.repmat(data.relations, 1, corrupt_size)), 
                    self.e2_holder        :  np.ravel(np.matlib.repmat(data.e2, 1, corrupt_size)),
                    self.e3_holder        :  data.e3Make,
                    self.treeLength_holder:  data.lens, 
                    self.tree_holder      :  data.out,
                    self.pred             :  data.flip
                }
            else:
                feeddict = {
                    self.e1_holder        :  np.ravel(np.matlib.repmat(data.e1[indexes], 1, corrupt_size)),
                    self.relation_holder  :  np.ravel(np.matlib.repmat(data.relations[indexes], 1, corrupt_size)), 
                    self.e2_holder        :  np.ravel(np.matlib.repmat(data.e2[indexes], 1, corrupt_size)),
                    self.e3_holder        :  data.e3Make,
                    self.treeLength_holder:  data.lens, 
                    self.tree_holder      :  data.out,
                    self.pred             :  data.flip
                }
                
        return feeddict;

    def update_x_2(self, inputVar):
        return inputVar;

    # forward run 
    def makeComputeGraph(self):
        self.e1_holder         = self.makeHolderInt();
        self.e2_holder         = self.makeHolderInt();
        self.relation_holder   = self.makeHolderInt();
        self.e3_holder         = self.makeHolderInt();    # change above too
        self.tree_holder       = self.makeHolderInt2D();
        self.treeLength_holder = self.makeHolderFloat();
        self.pred              = self.makeHolderBool();

        E_Var = tf.Variable(dtype=tf.float64, initial_value=self.E_matrix,trainable=True)

        with tf.control_dependencies([E_Var[:,0].assign(tf.zeros([100,], dtype = tf.float64))]):
            E_Var = tf.identity(E_Var);

        #W1_shape = [embedding_size, embedding_size, slice_size, data.num_relations]; # change num_relations pos
        #W1 = tf.Variable(tf.ones(shape=W1_shape, dtype = tf.float64)); #trun
        #W1 = tf.Variable(tf.truncated_normal(shape=W1_shape, dtype = tf.float64, stddev = 6.0 / embedding_size)); #trun
        W1 = tf.Variable(dtype=tf.float64, initial_value= W1Mat,trainable=True)
        #W2_shape = [data.num_relations, embedding_size * 2, slice_size]; 
        #W2 = tf.Variable(tf.ones(shape=W2_shape, dtype = tf.float64));
        #W2 = tf.Variable(tf.random_uniform(shape=W2_shape, dtype = tf.float64));   #randuni
        #W2_1_shape = [self.num_relations, embedding_size * 2, embedding_size * 2]; 
        #W2_1 = tf.Variable(tf.truncated_normal(shape=W2_1_shape, dtype = tf.float64, stddev = 6.0 / embedding_size));
        W2_2 = tf.Variable(dtype=tf.float64, initial_value= W2Mat,trainable=True)
        # b1 and u are extremely simple things
        #b1_1_shape = [self.num_relations, 1, embedding_size * 2,];
        #b1_1       = tf.Variable(tf.zeros(shape=b1_1_shape, dtype = tf.float64));   # grad of this var is awkward
        b1_2_shape = [self.num_relations, 1, slice_size,];
        b1_2       = tf.Variable(tf.zeros(shape=b1_2_shape, dtype = tf.float64)); 

        U_shape  = [self.num_relations, 1, slice_size,];        # U shape is different
        U        = tf.Variable(tf.ones(shape=U_shape, dtype = tf.float64));

        cost        = tf.Variable(0, dtype = tf.float64)
        scorePosNet = tf.Variable(tf.zeros(shape = [0,1], dtype = tf.float64));
        batchSize   = tf.constant(batch_size, dtype = tf.float64)
        reg_param   = tf.constant(0.0001, dtype = tf.float64)
        #x2 = tf.Variable([5])
        treeLengths = tf.reshape(self.treeLength_holder, [no_of_entities,1]);
        print self.treeLength_holder;
        Emat = tf.transpose(E_Var);
        collectedVectors = tf.gather(Emat, self.tree_holder);
        sumVecs = tf.reduce_sum(collectedVectors, axis = 1);
        print treeLengths;
        entVec1 = tf.divide(sumVecs,treeLengths);
        entVec = entVec1;


        # look to eliminate
        for i in xrange(self.num_relations):        

            lst = tf.where(tf.equal(self.relation_holder, i))
            e1 = tf.gather(self.e1_holder,lst);
            e2 = tf.gather(self.e2_holder,lst);
            e3 = tf.gather(self.e3_holder,lst);
            entVecE1 = tf.squeeze(tf.gather(entVec,e1));
            entVecE2 = tf.squeeze(tf.gather(entVec,e2));
            entVecE3 = tf.squeeze(tf.gather(entVec,e3));
            #entVecE1N = tf.Variable(tf.zeros(shape = entVecE1.get_shape()));

            entVecE1Neg = tf.cond(self.pred, lambda: self.update_x_2(entVecE1), lambda: self.update_x_2(entVecE3))
            entVecE2Neg = tf.cond(self.pred, lambda: self.update_x_2(entVecE3), lambda: self.update_x_2(entVecE2))

            W1transpose = tf.transpose(W1, perm=[3,2,0, 1]);
            # restrict yourself to special i
            W1specificTranspose = tf.gather(W1transpose,i);

            firstBi = tf.tensordot(W1specificTranspose, tf.transpose(entVecE2), axes = [[2], [0]]);
            firstBiNeg = tf.tensordot(W1specificTranspose, tf.transpose(entVecE2Neg), axes = [[2], [0]]);
            secondB = tf.multiply(tf.transpose(entVecE1), firstBi);
            secondBNeg = tf.multiply(tf.transpose(entVecE1Neg), firstBiNeg);
            finalBi = tf.reduce_sum(secondB , 1);
            finalBiNeg = tf.reduce_sum(secondBNeg , 1);

            #W2specific_1 = W2_1[i,:,:];
            #b1specific_1 = b1_1[i,:,:];
            W2specific_2 = W2_2[i,:,:];
            b1specific_2 = b1_2[i,:,:];

            concatEntVecs    = tf.concat([entVecE1, entVecE2], 1);
            concatEntNegVecs = tf.concat([entVecE1Neg, entVecE2Neg], 1);

            #simpleProd_1    = tf.tanh(tf.add(tf.matmul(concatEntVecs, W2specific_1), b1specific_1));
            #simpleNegProd_1 = tf.tanh(tf.add(tf.matmul(concatEntNegVecs, W2specific_1), b1specific_1));


            simpleProd_2    = tf.add(tf.matmul(concatEntVecs, W2specific_2), b1specific_2);
            simpleNegProd_2 = tf.add(tf.matmul(concatEntNegVecs, W2specific_2), b1specific_2);

            v_pos = simpleProd_2      + tf.transpose(finalBi);
            v_neg = simpleNegProd_2   + tf.transpose(finalBiNeg);
            z_pos = tf.tanh(v_pos);
            z_neg = tf.tanh(v_neg);

            UtransposeSpecific = tf.transpose(U[i,:,:]);
            score_pos          = tf.matmul(z_pos, UtransposeSpecific);
            score_neg          = tf.matmul(z_neg, UtransposeSpecific);

            bias = tf.constant(1, dtype = tf.float64);      # shold it be costant probably


            indx = tf.where(tf.greater(score_pos + bias, score_neg))
            indxJogar = tf.gather(tf.transpose(indx), 0);
            scorePosRel = tf.gather(score_pos, tf.transpose(indxJogar));
            scoreNegRel = tf.gather(score_neg, tf.transpose(indxJogar));
            partCost = tf.reduce_sum((scorePosRel + bias) - scoreNegRel);
            grads1 = tf.gradients(partCost, UtransposeSpecific);

            scorePosNet =  tf.concat([scorePosNet, score_pos], 0);  # required for test part

            cost = cost + partCost;

        squareSum = tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2_2)) + tf.reduce_sum(tf.square(b1_2));
        squareSum = squareSum +  tf.reduce_sum(tf.square(E_Var)) + tf.reduce_sum(tf.square(U));

        loss = tf.divide(cost,batchSize) #+ reg_param / 2.0 * squareSum;    # This division probably results in div
                                                                            # of gradients
        gradsE  = tf.gradients(loss, E_Var);
        gradsEmat  = tf.gradients(loss, Emat);
        gradsEntVec  = tf.gradients(loss, sumVecs);
        gradsW1 = tf.gradients(loss, W1);
        #gradsW2 = tf.gradients(loss, W2_1);
        #gradsB1 = tf.gradients(loss, b1_1);
        gradsU  = tf.gradients(loss, U);
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)  

        self.train_op = train_op;
        self.loss     = loss;

        return gradsEntVec, e1,scorePosNet, loss, train_op;


    def buildGraph(self):
        gradsEntVec, e1,scorePosNet, loss, train_op = self.makeComputeGraph()
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        
        return merged, e1,scorePosNet, loss, train_op

    def saveOps(self,savePath1,sess):
        path = savePath1 + 'Freebase_Logs/' + time.strftime("%Y-%m-%d-%H-%M-%S") 
        train_writer =  tf.summary.FileWriter(path + '/train/', sess.graph)
        test_writer  =  tf.summary.FileWriter(path + '/test/', sess.graph)

        saver = tf.train.Saver()
        return train_writer, test_writer, saver

    def evaluate(self, data, accuracy, sess, learningRate, indexes = ""):
        feeddict_new = self.makeFeedDict(data, learningRate, indexes);
        accRet= sess.run(accuracy, feeddict_new)
        return accRet

    def train(self, session, data):
        dataRows = len(data.e1)
        indexes = np.random.randint(0,dataRows,size = batch_size)
        if (random.uniform(0, 1) > 0.5):
            data.flip   = True;
        else:
            data.flip   = False;

        
        #flip   = True;
    
        feeddict_new = self.makeFeedDict(data, indexes, 10); # indexes or lstMat
        _,lossRet = session.run([self.train_op, self.loss] , feeddict_new);   # first Neg is wrong
        print 'loss', lossRet;
        return;