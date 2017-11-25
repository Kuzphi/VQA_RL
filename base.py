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
from torch.nn import Module, Linear, GRUCell, Embedding, Softmax, DataParallel


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
	parser.add_argument('--model_save',type=str, default='../model/',help='path of question data')
	# Training parameter
	parser.add_argument('--word_dim',        type=int,   default=300,  help='word feature size (default: 300)')
	parser.add_argument('--image_dim',       type=int,   default=2048, help='image feature size (default: 2048)')
	parser.add_argument('--classify_dim',      type=int,   default=2,    help='output dimension (default: 2)')
	
	# parser.add_argument('--decay_factor',    type=float, default=0.99997592083, help='decay factor of learning rate')
	parser.add_argument('--Classifier_lr', type=float, default=1e-5,help='Classifier Learning Rate (default: 1e-5)')
	parser.add_argument('--Classifier_itr', type=int, default=100,help='Classifier Iteration (default: 18)')

	parser.add_argument('--RNN_input_dim',  type=int,   default=300,  help='RNN_input_size (default: 300)')
	parser.add_argument('--RNN_output_dim',  type=int,   default=512,  help='RNN_input_size (default: 300)')
	parser.add_argument('--RNN_emb_dim', type=int,   default=512,  help='RNN_hidden_size (default: 512)')	
	parser.add_argument('--trans_dim', type=int, default=500, help='ValueNN Embedding trans dimension  (default: 500)')

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
	def __init__(self, Embed_matrix, **args):
		super(Model, self).__init__()
		self.emb_dim    = args['RNN_emb_dim']
		self.QuesGRU    = DataParallel( nn.GRU(input_size = args['RNN_input_dim'], hidden_size = args['RNN_emb_dim'], num_layers = 1, batch_first = 1).double())
		self.AnsGRU     = DataParallel( nn.GRU(input_size = args['RNN_input_dim'], hidden_size = args['RNN_emb_dim'], num_layers = 1, batch_first = 1).double())
		self.Qtrans 	= DataParallel( Linear(args['RNN_output_dim'] , args['trans_dim']).double()    )
		self.Atrans 	= DataParallel( Linear(args['RNN_output_dim'] , args['trans_dim']).double())
		self.Itrans 	= DataParallel( Linear(args['image_dim']      , args['trans_dim']).double())
		self.QItrans 	= DataParallel( Linear(args['trans_dim']		, args['trans_dim']).double())
		self.QIAtrans 	= DataParallel( Linear(args['trans_dim']		, args['trans_dim']).double())
		self.classifier = DataParallel( Linear(args['trans_dim']	    , args['classify_dim']).double())
		self.Embed_lookup = Embedding(*Embed_matrix.shape).double()
		self.Embed_lookup.weight.data.copy_(from_numpy(Embed_matrix).double())
		self.Embed_lookup.weight.requires_grad = False

	def forward(self, Img, Ques, Ans):
		_Ans  = self.Embed_lookup(Ans)
		_Ques = self.Embed_lookup(Ques)

		bs = Img.shape[0]
		Qh0 = Variable(torch.randn(1, bs, self.emb_dim)).cuda().double()
		Q, Qh = self.QuesGRU(_Ques, Qh0)
		Q = self.Qtrans(Q[:, -1, :])

		Ah0 = Variable(torch.randn(1, bs, self.emb_dim)).cuda().double()
		A, Ah = self.AnsGRU(_Ans, Ah0)
		A = self.Atrans(A[:, -1, :])

		I = self.Itrans(Img.mean(dim = 1))
		QI = self.QItrans(Q * I)

		QIA = self.QIAtrans(QI * A)
		confi = F.softmax(self.classifier(QIA), dim = 1)
		return confi

def Valid(Model, Img, data, bs, **kwargs):
	print ("\tValiding")
	record, res= {}, {}
	for index, img, ques, ans, targets in tqdm(Classifier_batch_generator(Img, data, bs, 3)):
		confidences = Model.forward(	Variable(from_numpy(img), volatile = True).cuda(),
								Variable(from_numpy(ques).long(), volatile = True).cuda(),
								Variable(from_numpy(ans).long() , volatile = True).cuda() 
							).cpu()
		for id, confidence, target in zip(index, confidences, targets):
			# print id, confidence, target
			# print (id // 4, id, confidence[0], target[0])
			if (id // 4 not in record) or confidence[0] > record[id // 4]:
				record[id // 4] = confidence[0]
				res[id // 4] = target[0]

	acc = sum(res.values()) / len(record)
	return acc
	


def Train(global_itr, Classifier_itr, Classifier_lr, model_save, **args):
	train_img, train_data, test_img, test_data, emb_matrix = Get_Data(**args)
	print("AAAAA")
	num_train = train_data['question'].shape[0]
	model   = Model(Embed_matrix = emb_matrix, **args).cuda()
	print("construct model")
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = Classifier_lr)
	print("construct optimizer")
	# Valid(model, test_img, test_data, 8, **args)
	print("start training");
	best = 0
	for itr in range(Classifier_itr):
		print ("Iteration %d :"%(itr))
		Losses = []
		for _, img, ques, ans, target in tqdm(Classifier_batch_generator(train_img, train_data, 18, 1)):
			optimizer.zero_grad()
			confi = model.forward(	Variable(from_numpy(img)).cuda(),
									Variable(from_numpy(ques).long()).cuda(),
									Variable(from_numpy(ans).long()).cuda()).cpu()
			loss = F.binary_cross_entropy(confi, Variable(from_numpy(target)).double())
			loss.backward()
			optimizer.step()
			Losses.append(loss.data.numpy())
		print("\tTraining Loss : %.3f"%(np.array(Losses).mean()))
		if itr and itr % 5 == 0:
			acc = Valid(model, test_img, test_data, 8, **args)
			print ('\tAccuracy of test: %.4f'%(acc))
			if acc > best:
				best = acc
				torch.save({
					'state_dict': model.state_dict(),
					'optimizer' : optimizer.state_dict(),
					'parameters': gargs
					}, model_save + str(best))

if __name__ == '__main__':
	# torch.multiprocessing.set_start_method("spawn")
	global gargs
	gargs = Arguement()
	# torch.setdefaulttensortype('torch.DoubleTensor')
	# with torch.cuda.device(3):
	Train(**gargs)
