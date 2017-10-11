import numpy as np 
from scipy.sparse import csr_matrix
embedding_size = 100;
slice_size = 3;


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def tanh(x):
	return np.tanh(x)

def tanh_deriv(x):
	return (1.0 - x**2);	# We assume that the tanh as already been taken
#from itertools import zip_longest

def tensorCost(data, W1, W2, b1, U, tree, flipType):
	cost = 0;
	update = 0;
	gradb1 = np.zeros(shape = b1.shape);
	gradW1 = np.zeros(shape = W1.shape);
	gradW2 = np.zeros(shape = W2.shape);
	gradU  = np.zeros(shape = U.shape);
	entVecGrad = np.zeros(shape = (embedding_size, data.entity_length));

	entVec = np.zeros(shape = (embedding_size, data.entity_length));
	for i in xrange(data.entity_length):
		ids = tree[i];
		ids = np.asarray(map(int, ids)) - 1; # matlab to python index
		entVec[:,i] = np.mean(data.word_embeds[:, ids], 1);
	#print ids, i
	for i in xrange(data.num_relations):
		lst = (data.relations == i);
		print len(lst);

		mAll = sum(lst);
		print mAll;
		e1 = data.e1[lst];
		e2 = data.e2[lst];
		e3 = data.e3[lst];

		entVecE1 = entVec[:, e1];
		entVecE2 = entVec[:, e2];
		entVecE3 = entVec[:, e3];

		if flipType:
			# replacing right entities with random ones
			entVecE1Neg = entVecE1;
			entVecE2Neg = entVecE3;
			e1Neg = e1;
			e2Neg = e3;
		else:
			# replacing left entities with random ones
			entVecE1Neg = entVecE3;
			entVecE2Neg = entVecE2;
			entVecE1Neg = entVecE1Neg.reshape(entVecE1Neg.shape[:2])
			entVecE2Neg = entVecE2Neg.reshape(entVecE2Neg.shape[:2])

			e1Neg = e3;
			e2Neg = e2;

		v_pos = np.zeros(shape = (slice_size, mAll))



		for k in xrange(slice_size):
			v_pos[k, :] = sum(np.multiply(entVecE1,np.dot(W1[:, :, k, i], entVecE2)));

		simpleProd = np.dot(np.transpose(W2[:, :, i]), np.concatenate((entVecE1, entVecE2), axis=0)) + np.transpose(b1[:, :, i]);
		v_pos = v_pos + simpleProd;

		v_neg = np.zeros(shape=(slice_size, mAll))
		for k in xrange(slice_size):
			v_neg[k, :] = sum(np.multiply(entVecE1Neg,np.dot(W1[:, :, k, i], entVecE2Neg)));

		simpleProd = np.dot(np.transpose(W2[:, :, i]), np.concatenate((entVecE1Neg, entVecE2Neg), axis=0)) + np.transpose(b1[:, :, i]);
		v_neg = v_neg + simpleProd;
			#print 'd', d[0];

		z_pos = tanh(v_pos);
		z_neg = tanh(v_neg);
		score_pos = np.dot(np.transpose(U[:,:,i]), z_pos);
		score_neg = np.dot(np.transpose(U[:, :, i]), z_neg);

		#print 'z_pos', z_pos[0];
		#print 'score_pos', score_pos[0];

		indx = (score_pos + 1 > score_neg);
		print indx
		c = np.multiply(indx , (score_pos + 1 - score_neg));
		print 'mulitply',c.shape
		a = np.sum(np.multiply(indx , (score_pos + 1 - score_neg)));
		cost = cost + np.sum(np.multiply(indx , (score_pos + 1 - score_neg)));

		print 'cost', cost;

		# filter for only active
		
		indx = indx.ravel();


		m = np.sum(indx);
		z_pos=z_pos[:,indx];
		z_neg=z_neg[:, indx];
		entVecE1Rel = entVecE1[:,indx];
		entVecE2Rel = entVecE2[:,indx];
		entVecE1RelNeg = entVecE1Neg[:, indx];
		entVecE2RelNeg = entVecE2Neg[:, indx];
		e1 = e1[indx];
		e2 = e2[indx];
		e1Neg = e1Neg[indx];
		e2Neg = e2Neg[indx];
		e1Neg = e1Neg.ravel();
		e2Neg = e2Neg.ravel();


		gradU[:,:,i] = (np.sum(z_pos - z_neg,1))[:,None];

		deriv_pos = tanh_deriv(z_pos);
		deriv_neg = tanh_deriv(z_neg);

		tmp_posAll = np.multiply(U[:,:,i], deriv_pos);
		tmp_negAll = -1 * np.multiply(U[:,:,i], deriv_neg);
		gradb1[:,:,i]  = np.sum(tmp_posAll + tmp_negAll,1);


		for k in xrange(slice_size):
			tmp_pos = tmp_posAll[k,:];
			tmp_neg = tmp_negAll[k, :];
			
			gradW1[:, :, k,i] = np.dot(np.multiply(entVecE1Rel, tmp_pos) , np.transpose(entVecE2Rel))
			gradW1[:, :, k, i] = gradW1[:, :, k, i] + np.dot(np.multiply(entVecE1RelNeg, tmp_neg), np.transpose(entVecE2RelNeg))

			gradW2[:, k, i] = np.sum(np.multiply(np.concatenate((entVecE1Rel, entVecE2Rel),axis = 0), tmp_pos),1)
			gradW2[:, k, i] = gradW2[:, k, i] + np.sum(np.multiply(np.concatenate((entVecE1RelNeg, entVecE2RelNeg), axis=0), tmp_neg), 1)
			
			W2special = W2[:, k, i]; # magic step
			V_pos = np.multiply(W2special[:,None], tmp_pos[None, :]);	# RISKY
			V_neg = np.multiply(W2special[:, None], tmp_neg[None, :]);

			# from the especial tensor product
			e1Sparse    = csr_matrix((np.ones(shape = (m)), (range(m), e1)), shape=(m, data.entity_length));
			e2Sparse    = csr_matrix((np.ones(shape = (m)), (range(m), e2)), shape=(m, data.entity_length));
			e1NegSparse = csr_matrix((np.ones(shape = (m)), (range(m), e1Neg)), shape=(m, data.entity_length));
			e2NegSparse = csr_matrix((np.ones(shape = (m)), (range(m), e2Neg)), shape=(m, data.entity_length));

			e1SparseT = e1Sparse.transpose();
			e2SparseT = e2Sparse.transpose();
			e1NegSparseT = e1NegSparse.transpose();
			e2NegSparseT = e2NegSparse.transpose();


			e1Prod = e1SparseT.dot(np.transpose(V_pos[0:embedding_size,:]));
			e2Prod = e2SparseT.dot(np.transpose(V_pos[0:embedding_size,:]));
			e1NegProd = e1NegSparseT.dot(np.transpose(V_neg[0:embedding_size,:]));
			e2NegProd = e2NegSparseT.dot(np.transpose(V_neg[0:embedding_size,:]));

			entVecGrad = entVecGrad + np.transpose(e1Prod) + np.transpose(e2Prod) + np.transpose(e1NegProd) + np.transpose(e2NegProd);

			# from seperate entities at a time
			e2TensorGrad = np.multiply(np.dot(W1[:,:,k,i], entVec[:,e2]), tmp_pos);
			e1TensorGrad = np.multiply(np.dot(W1[:,:,k,i], entVec[:,e1]), tmp_pos);
			e2NegTensorGrad = np.multiply(np.dot(W1[:,:,k,i], entVec[:,e2Neg]), tmp_neg);
			e1NegTensorGrad = np.multiply(np.dot(W1[:,:,k,i], entVec[:,e1Neg]), tmp_neg);
			e1Prod = e1SparseT.dot(np.transpose(e2TensorGrad));
			e2Prod = e2SparseT.dot(np.transpose(e1TensorGrad));
			e1NegProd = e1NegSparseT.dot(np.transpose(e2NegTensorGrad));
			e2NegProd = e2NegSparseT.dot(np.transpose(e1NegTensorGrad));

			entVecGrad = entVecGrad + np.transpose(e1Prod) + np.transpose(e2Prod) + np.transpose(e1NegProd) + np.transpose(e2NegProd);
			


		gradW1[:,:,:,i] = gradW1[:,:,:,i] / len(data.relations);
		gradb1[:,:,i]   = gradb1[:,:,i]   / len(data.relations);
		gradU[:,:,i]    = gradU[:,:,i]    / len(data.relations);
		gradW2[:,:,i]   = gradW2[:,:,i]   / len(data.relations);
	
		
	gradE = np.zeros(shape = data.word_embeds.shape);
	for e in xrange(data.entity_length):
		ids = tree[e];
		sl = len(ids);
		ids = np.asarray(map(int, ids)) - 1; # matlab to python index
		#print ids, e;
		addition = np.tile(entVecGrad[:, e] / sl, (sl, 1));
		gradE[:, ids] = gradE[:, ids] + np.transpose(addition);

		
	#print 'gradE', gradE.shape;
	#print 'gradE[0,0]', gradE[0:5,4658:4668];
	print 'cost', cost;
	gradE = gradE / len(data.relations);
	cost  = cost  / len(data.relations);

	reg_parameter = 0.0001;
	print 'cost', cost;
	squareSum = np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(b1)) + np.sum(np.square(U)) + np.sum(np.square(data.word_embeds));
	cost = cost + reg_parameter / 2 * squareSum;

	gradW1 = gradW1 + reg_parameter * W1;
	gradW2 = gradW2 + reg_parameter * W2;
	gradb1 = gradb1 + reg_parameter * b1;
	gradU  = gradU  + reg_parameter * U;




	return cost;