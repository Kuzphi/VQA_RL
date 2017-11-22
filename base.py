from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy, optim
from torch.autograd import Variable
import pickle
import argparse
from tqdm import tqdm
from torch.nn import Module, Linear, GRUCell, Embedding, Softmax


from os import path
from copy import copy
from numpy import array, zeros
from misc import Get_Data, Classifier_batch_generator, ValueNN_batch_generator, load_ValueNN_data, fetch_records
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
	parser.add_argument('--RNN_output_dim',  type=int,   default=512,  help='RNN_input_size (default: 300)')
	parser.add_argument('--RNN_emb_dim', type=int,   default=512,  help='RNN_hidden_size (default: 512)')	
	parser.add_argument('--Classifier_lr', type=float, default=1e-5,help='Classifier Learning Rate (default: 1e-5)')
	parser.add_argument('--Classifier_bs', type=int, default=18,help='batch_size for each Classifier iterations (default: 18)')
	parser.add_argument('--Classifier_Trans_dim', type=int,  default=4096, help='default: 4096')
	# parser.add_argument('--Classifier_ImgTrans_dim', type=int,  default=4096, help='default: 4096')
	# parser.add_argument('--Classifier_AnsTrans_dim', type=int,  default=4096, help='default: 4096')
	# parser.add_argument('--Classifier_QuesTrans_dim', type=int, default=4096, help='default: 4096')
	parser.add_argument('--Classifier_itr', type=int, default=1,help='Classifier Iteration (default: 18)')
	
	parser.add_argument('--ValueNN_RNN2Value_dim', type=int, default=500, help='ValueNN Embedding trans dimension  (default: 500)')
	parser.add_argument('--ValueNN_Img2Value_dim', type=int, default=500, help='ValueNN Images trans dimension (default: 500)')
	parser.add_argument('--ValueNN_itr', type=int, default=50, help='ValueNN training maximum Iteration  (default: 10)')
	parser.add_argument('--ValueNN_lr', type=float, default=1e-5,help='ValueNN Learning Rate (default: 18)')
	parser.add_argument('--ValueNN_bs', type=int, default=18,help='batch_size for each ValueNN iterations (default: 18)')

	parser.add_argument('--MC_RollOut_Size', type=int, default=3000, help='default: 4096')
	parser.add_argument('--MC_Root_size', type=int, default=1,help='default: 1000')



	#Misc
	parser.add_argument('--ValueNNDataPath', type=str, default="../data/ValueNN/", help='path for storing ValueNN train data')
	parser.add_argument('--max_words_q', type=int, default=26, help='maximum words for a sentence(default: 26)')
	parser.add_argument('--global_itr', type=int, default=1000, help='maximum iteration(default: 1000)')
	parser.add_argument('--epoches', type=int, default=300, help='epoches(default: 300)')
	parser.add_argument('--seed', type=int, default=320, help='random seed (default: 320)')
	#GPU
	parser.add_argument('--cuda', action='store_true', default=True,help='enable CUDA training')
	return vars(parser.parse_args())


class Model(Module):
	def __init__(self, **args):
		super(ValueNN_Module, self).__init__()
		self.emb_dim    = args['RNN_emb_dim']
		self.QuesGRU    = GRU(input_size = args['RNN_input_dim'], hidden_size = args['RNN_emb_dim'], batch_first = 1)
		self.AnsGRU     = GRU(input_size = args['RNN_input_dim'], hidden_size = args['RNN_emb_dim'], batch_first = 1)
		self.Qtrans 	= Linear(args['RNN_output_dim'], args['trans_dim'])
		self.Atrans 	= Linear(args['RNN_output_dim'], args['trans_dim'])
		self.Itrans 	= Linear(args['image_dim']     , args['trans_dim'])
		self.QItrans 	= Linear(args['RNN_output_dim'], args['trans_dim'])
		self.QIAtrans 	= Linear(args['RNN_output_dim'], args['trans_dim'])
		self.classifier = Linear(args['trans_dim']	   , args['classify_dim'])
	def forward(self, Img, Ques, Ans):
		bs = I.shape[0]
		Qh0 = Variable(torch.randn(1, bs, self.emb_dim))
		Q, Qh = self.QuesGRU(Ques, Qh0)
		Q = self.Qtrans(Q[-1,:,1])

		Ah0 = Variable(torch.randn(1, bs, self.emb_dim))
		A, Ah = self.AnsGRU(Ans, Ah0)
		A = self.Atrans(A[-1,:,1])

		I = self.Itrans(I.mean(dim = 1))
		QI = self.QItrans(Q * I)

		QIA = self.QIAtrans(QI * A)
		confi = F.softmax(self.classifier(QIA), dim = 1)
		return confi

def Valid(Model, Img, data, bs, **kwargs):
	print ("\tValiding")
	record = {}
	for index, img, ques, ans, len_a, targets in tqdm(Classifier_batch_generator(Img, data, bs, 3)):
		confi = Model.forward(Variable(Img), Varibale(ques), Variable(ans))
		for id, confidence, target in zip(index, confidences, targets):
			# print id, confidence, target
			if (not record.has_key(id / 4)) or confidence[0] > record[id / 4][0]:
				record[id / 4] = [confidence[0], target[0]]

	acc = 1.0 * array(record.values())[:,1].sum() / len(record)
	print ('\tAccuracy of test: %d'%(acc))


def Train(global_itr, **args):
	train_img, train_data, test_img, test_data, emb_matrix = Get_Data(**args)
	num_train = train_data['question'].shape[0]

	model   = Model(**args).cuda()

	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = Classifier_lr)
	for itr in range(Classifier_itr):
		print ("Iteration %d :")
		Losses = []
		for _, img, ques, ans, len_a, target in tqdm(Classifier_batch_generator(imgs, data, Classifier_bs, 1)):
			optimizer.zero_grad()
			confi = model.forward(img,ques,ans)
			loss = F.binary_cross_entropy(confidence, Variable(from_numpy(target)))
			loss.backward()
			optimizer.step()
			Losses.append(loss.data.numpy())
		print("\tTraining Loss : %.3f"%(np.array(Losses).mean()))
		if itr % 10 == 0:
			print("\tValid Accuracy : %.3f"%( Valild(Model, test_img, test_data, 20) ))

if __name__ == '__main__':
	# torch.multiprocessing.set_start_method("spawn")
	args = Arguement()
	# torch.setdefaulttensortype('torch.DoubleTensor')
	# with torch.cuda.device(3):
	Train(**args)
