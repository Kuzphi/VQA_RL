import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from data import Get_Data
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
	parser.add_argument('--word_dim', type=int, default=300,help='word feature size (default: 300)')
	parser.add_argument('--image_dim', type=int, default=2048,help='image feature size (default: 2048)')
	parser.add_argument('--num_output', type=int, default=2, help='num of output dimension (default: 2)')
	parser.add_argument('--decay_factor', type=float, default=0.99997592083, help='decay factor of learning rate')
	parser.add_argument('--RNN_input_size', type=int, default=300, help='RNN_input_size (default: 300)')
	parser.add_argument('--RNN_hidden_size', type=int, default=512, help='RNN_hidden_size (default: 512)')
	parser.add_argument('--Classfier_ImgTrans_dim', type=int, default=4096, help='RNN_hidden_size (default: 512)')
	parser.add_argument('--Classfier_AnsTrans_dim', type=int, default=4096, help='RNN_hidden_size (default: 512)')
	parser.add_argument('--Classfier_WordTrans_dim', type=int, default=4096, help='RNN_hidden_size (default: 512)')
	

	#Misc
	parser.add_argument('--max_words_q', type=int, default=26, help='maximum words for a sentence(default: 26)')
	parser.add_argument('--max_itr', type=int, default=1000, help='maximum iteration(default: 1000)')
	parser.add_argument('--epoches', type=int, default=300, help='epoches(default: 300)')
	parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
	#GPU
	parser.add_argument('--cuda', action='store_true', default=True,help='enable CUDA training')
	return vars(parser.parse_args())
def Train_Classifier():

def Train_ValueNN():

def Generate_ValueNN_train_date():
	
def Valid():

def Train(**kwargs):
	train_img, train_data, test_img, test_data, emb_matrix = Get_Data(arg)
	num_train = train_data['question'].shape[0]
	ValueNN   = ValueNN(args)
	Classifer = Classifer(args)
	for itr in global_max_itr:
		Generate_ValueNN_train_data(Classifer)
		Train_ValueNN(Classfier, ValueNN, train_img, train_data)
		Train_Classifier(Classfier, ValueNN, train_img, train_data)
		Valid(test_img, test_data, emb_matrix)


if __name__ == '__main__':
	args = Arguement()
	Train()



