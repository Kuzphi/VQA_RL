import np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.optim as optim
import argparse
from copy import copy
from numpy import array, zeros
from Misc import Get_Data, batch_generator
from Model import ValueNN, Classfier, MonteCarloTree
def Arguement():
	parser = argparse.ArgumentParser()
	# Data Path
	parser.add_argument('--input_img_h5_train', type=str, default='spatial_data_img_residule_train_14by14to7by7_norm.h5',
						help='path of train image feature')
	parser.add_argument('--input_img_h5_test', type=str, default= 'spatial_data_img_residule_test_14by14to7by7_norm.h5', 
						help='path of testing image feature')
	parser.add_argument('--input_ques_h5', type=str, default='./data_prepro_0417_v1.h5',
						help='path of question data')
	# Training parameter
	parser.add_argument('--Classfier_lr', type=float, default=1e-5,help='Classfier Learning Rate (default: 18)')
	parser.add_argument('--valueNN_lr', type=float, default=1e-5,help='ValueNN Learning Rate (default: 18)')
	parser.add_argument('--Classfier_itr', type=int, default=18,help='Classfier Iteration (default: 18)')
	parser.add_argument('--ValueNN_itr', type=int, default=18,help='ValueNN Iteration (default: 18)')
	parser.add_argument('--batch_size', type=int, default=18,help='batch_size for each iterations (default: 18)')
	parser.add_argument('--word_dim', type=int, default=300,help='word feature size (default: 300)')
	parser.add_argument('--image_dim', type=int, default=2048,help='image feature size (default: 2048)')
	parser.add_argument('--num_output', type=int, default=2, help='num of output dimension (default: 2)')
	parser.add_argument('--decay_factor', type=float, default=0.99997592083, help='decay factor of learning rate')
	parser.add_argument('--RNN_input_size', type=int, default=300, help='RNN_input_size (default: 300)')
	parser.add_argument('--RNN_hidden_size', type=int, default=512, help='RNN_hidden_size (default: 512)')
	parser.add_argument('--Classfier_ImgTrans_dim', type=int, default=4096, help='default: 4096')
	parser.add_argument('--Classfier_AnsTrans_dim', type=int, default=4096, help='default: 4096')
	parser.add_argument('--Classfier_WordTrans_dim', type=int, default=4096, help='default: 4096')
	parser.add_argument('--MonteCarloGenerateSize', type=int, default=3000, help='default: 4096')
	#Misc
	parser.add_argument('--max_words_q', type=int, default=26, help='maximum words for a sentence(default: 26)')
	parser.add_argument('--max_itr', type=int, default=1000, help='maximum iteration(default: 1000)')
	parser.add_argument('--epoches', type=int, default=300, help='epoches(default: 300)')
	parser.add_argument('--seed', type=int, default=320, help='random seed (default: 320)')
	#GPU
	parser.add_argument('--cuda', action='store_true', default=True,help='enable CUDA training')
	return vars(parser.parse_args())

def Train_Classifier(**kwargs):
	learning_rate = lr
	optimizer = optimer.adam(Classfier.parameter(), lr =lr)
	for itr in xrange(classfier_itr):
		for _, img, ques, ans, len_a, target in batch_generator(Imgs, data, batch_size):
			confidence, _ = Classfier.forward(ques, ans, ValueNN, img)
			loss = F.binary_cross_entropy(confidence, target)
			loss.backward()
			optimizer.step()
			if itr % 1000 == 0:
				print "Classifer Iteration " + str(itr) + ": " + str(loss.data)




def Train_ValueNN(**kwargs):
	imgs, states, targets = load_ValueNN_data(global_itr)
	optimizer = optimer.adam(ValueNN.parameter(), lr =lr)

	for itr in xrange(ValueNN_itr):
		for _, img, state, target in zip(imgs, states, targets)
			RNN_emb = array([copy(state) for i in xrange(len(target))])
			value = ValueNN.forward(RNN_emb, img)
			loss = F.mseloss(value, target)
			loss.backward()
			optimizer.step()
			if itr % 1000 == 0:
				print "ValueNN Iteration " + str(itr) + ": " + str(loss.data)

def Generate_MoteCarloTree_Root(Img, Data, Classfier,itr, maximum):
	record = []
	cnt = 0
	for index, img, ques, ans, len_a, targets in batch_generator(Imgs, data, 1):
		_, states = Classfier.forward(ques, ans, ValueNN, img)
		step = np.random.randint(0,25)
		record.append([index, id, states[:,id,:], targets])
		cnt += 1
		if cnt > maximum: break;
	return record
def Generate_ValueNN_train_data(records):
	for index, id, state in records:
		xid = np.random.randint(0.4)
		MonteCarloTree()
		new_sample 

def Valid(Img, test_data,Classfier, ValueNN):
	record = zeros(len(test_data), 2)
	for index, img, ques, ans, len_a, targets in batch_generator(Imgs, data, batch_size):
		confidences, _ = Classfier.forward(ques, ans, ValueNN, img)
		for id, confidence, target in zip(index, confidences, targets):
			if confidence > record[id / 4]:
				record[id / 4][0] = confidence
				record[id / 4][1] = target[0]
	acc = record[:,1] * test_data['target'][:,0]
	return acc


def Train(**kwargs):
	train_img, train_data, test_img, test_data, emb_matrix = Get_Data(arg)
	num_train = train_data['question'].shape[0]
	ValueNN   = ValueNN(kwargs)
	Classifer = Classifer(emb_matrix, kwargs)
	for itr in global_max_itr:
		print "Iteration " + str(itr)
		record = Generate_MoteCarloTree_Root(train_img, train_data, Classifer, itr, MonteCarloGenerateSize)
		Generate_ValueNN_train_data(record)
		Train_ValueNN(Classfier, ValueNN, train_img, train_data, ValueNN_lr, ValueNN_itr)
		Train_Classifier(Classfier, ValueNN, train_img, train_data, Classfier_lr,Classfier_itr)
		accuracy = Valid(test_img, test_data, Classfier, ValueNN)
		print 'Accuracy of test: ' + str(acc*1.0/len(result))


if __name__ == '__main__':
	args = Arguement()
	Train(args)



