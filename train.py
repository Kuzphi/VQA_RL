import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.optim as optim
import pickle
import argparse
from os import path
from copy import copy
from numpy import array, zeros
from misc import Get_Data, Classifier_batch_generator, ValueNN_batch_generator
from model import ValueNN_Module, Classifier_Module, MonteCarloTree_Module
def Arguement():
	parser = argparse.ArgumentParser()
	# Data Path
	parser.add_argument('--train_img_h5', type=str, default='../data/spatial_data_img_residule_train_14by14to7by7_norm.h5',
						help='path of train image feature')
	parser.add_argument('--test_img_h5',  type=str, default='../data/spatial_data_img_residule_test_14by14to7by7_norm.h5', 
						help='path of testing image feature')
	parser.add_argument('--ques_h5',type=str, default='../data/data_prepro_0417_v1.h5',help='path of question data')

	# Training parameter
	parser.add_argument('--word_dim',        type=int,   default=300,  help='word feature size (default: 300)')
	parser.add_argument('--image_dim',       type=int,   default=2048, help='image feature size (default: 2048)')
	parser.add_argument('--classify_dim',      type=int,   default=2,    help='output dimension (default: 2)')
	
	# parser.add_argument('--decay_factor',    type=float, default=0.99997592083, help='decay factor of learning rate')
	
	parser.add_argument('--RNN_input_dim',  type=int,   default=300,  help='RNN_input_size (default: 300)')
	parser.add_argument('--RNN_emb_dim', type=int,   default=512,  help='RNN_hidden_size (default: 512)')	
	parser.add_argument('--Classifier_lr', type=float, default=1e-5,help='Classifier Learning Rate (default: 1e-5)')
	parser.add_argument('--Classifier_bs', type=int, default=18,help='batch_size for each Classifier iterations (default: 18)')
	parser.add_argument('--Classifier_ImgTrans_dim', type=int,  default=4096, help='default: 4096')
	parser.add_argument('--Classifier_AnsTrans_dim', type=int,  default=4096, help='default: 4096')
	parser.add_argument('--Classifier_QuesTrans_dim', type=int, default=4096, help='default: 4096')
	parser.add_argument('--Classifier_itr', type=int, default=18,help='Classifier Iteration (default: 18)')
	
	parser.add_argument('--ValueNN_RNN2Value_dim', type=int, default=500, help='ValueNN Embedding trans dimension  (default: 500)')
	parser.add_argument('--ValueNN_Img2Value_dim', type=int, default=500, help='ValueNN Images trans dimension (default: 500)')
	parser.add_argument('--ValueNN_itr', type=int, default=10, help='ValueNN training maximum Iteration  (default: 10)')
	parser.add_argument('--valueNN_lr', type=float, default=1e-5,help='ValueNN Learning Rate (default: 18)')
	parser.add_argument('--ValueNN_bs', type=int, default=18,help='batch_size for each ValueNN iterations (default: 18)')

	parser.add_argument('--MC_RollOut_Size', type=int, default=3000, help='default: 4096')
	parser.add_argument('--MC_Root_size', type=int, default=1000,help='1000')



	#Misc
	parser.add_argument('--ValueNNDataPath', type=str, default="../data/ValueNN/", help='path for storing ValueNN train data')
	parser.add_argument('--max_words_q', type=int, default=26, help='maximum words for a sentence(default: 26)')
	parser.add_argument('--global_itr', type=int, default=1000, help='maximum iteration(default: 1000)')
	parser.add_argument('--epoches', type=int, default=300, help='epoches(default: 300)')
	parser.add_argument('--seed', type=int, default=320, help='random seed (default: 320)')
	#GPU
	parser.add_argument('--cuda', action='store_true', default=True,help='enable CUDA training')
	return vars(parser.parse_args())

def Generate_MoteCarloTree_Root(Classifier, ValueNN, Imgs, Data, itr, **kwargs):
	record = []
	cnt = 0
	for index, img, ques, ans, len_a, targets in Classifier_batch_generator(Imgs, Data, 1):
		_, states = Classifier.forward(ValueNN, img, ques, ans)
		step = np.random.randint(0,26)
		record.append([index, states[:,step,:], targets])
		cnt += 1
		if cnt > Root_size: break;
	return record

def Generate_ValueNN_train_data(Classifier, records, Img, Data, Itr, **kwargs):
	Record = []
	for index, id, state, targets in records:
		xid = np.random.randint(0,state.shape[0])
		Tree = MonteCarloTree_Module(state[xid], Data['Question'], Data['Answer'], Img[index], targets, Classifier)
		value = Tree.Generate(1000)
		Record.append([state[xid], value, index])
	Pickle.dump(Record, open(path.join(ValueNNDataPath,"ValueNN Train " + str(Itr)), 'w'))

def Train_ValueNN(ValueNN, img, global_itr, ValueNN_bs, ValueNNDataPath, **kwargs):
	imgs, states, targets = load_ValueNN_data(img, ValueNNDataPath, global_itr)
	optimizer = optimer.adam(ValueNN.parameter(), lr =lr)

	for itr in xrange(ValueNN_itr):
		for img, state, target in ValueNN_batch_generator(imgs, states, targets, ValueNN_bs):
			RNN_emb = array([copy(state) for i in xrange(len(target))])
			value = ValueNN.forward(RNN_emb, img)
			loss = F.mseloss(value, target)
			loss.backward()
			optimizer.step()
			if itr % 1000 == 0:
				print "ValueNN Iteration " + str(itr) + ": " + str(loss.data)

def Train_Classifier(Classifier, ValueNN, imgs, data, Classifier_bs, **kwargs):
	learning_rate = lr
	optimizer = optimer.adam(Classifier.parameter(), lr =lr)
	for itr in xrange(Classifier_itr):
		for _, img, ques, ans, len_a, target in Classifier_batch_generator(imgs, data, Classifier_bs):
			confidence, _ = Classifier.forward(ques, ans, ValueNN, img)
			loss = F.binary_cross_entropy(confidence, target)
			loss.backward()
			optimizer.step()
			if itr % 1000 == 0:
				print "Classifier Iteration " + str(itr) + ": " + str(loss.data)

def Valid(Classifier, ValueNN, Img, test_data, Classifier_bs, **kwargs):
	record = zeros(len(test_data), 2)
	for index, img, ques, ans, len_a, targets in Classifier_batch_generator(Imgs, data, Classifier_bs):
		confidences, _ = Classifier.forward(ques, ans, ValueNN, img)
		for id, confidence, target in zip(index, confidences, targets):
			if confidence > record[id / 4]:
				record[id / 4][0] = confidence
				record[id / 4][1] = target[0]
	acc = record[:,1] * test_data['target'][:,0]
	return acc

def Train(global_itr, **args):
	train_img, train_data, test_img, test_data, emb_matrix = Get_Data(**args)
	num_train = train_data['question'].shape[0]

	ValueNN   = ValueNN_Module(**args)
	Classifier = Classifier_Module(emb_matrix, **args)

	for itr in xrange(global_itr):
		print "Iteration " + str(itr)
		record = Generate_MoteCarloTree_Root(Classifier, ValueNN, train_img, train_data, itr, **args)
		Generate_ValueNN_train_data(Classifier, record, train_img, train_data, itr, **args)
		Train_ValueNN(ValueNN, train_img, itr, **args)
		Train_Classifier(Classifier, ValueNN, train_img, train_data, **args)
		accuracy = Valid(Classifier, ValueNN, test_img, test_data, **args)
		print 'Accuracy of test: ' + str(acc*1.0/len(result))


if __name__ == '__main__':
	args = Arguement()
	Train(**args)



