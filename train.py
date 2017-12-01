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

from os import path
from copy import copy
from numpy import array, zeros
from misc import Get_Data, Classifier_batch_generator, ValueNN_batch_generator, load_ValueNN_data, fetch_records
from model import ValueNN_Module, Classifier_Module, MonteCarloTree_Module
def Arguement():
	parser = argparse.ArgumentParser()
	#Path
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
	
	parser.add_argument('--RNN_input_dim',  type=int,   default=300 + 2048,  help='RNN_input_size (default: 300)')
	parser.add_argument('--RNN_output_dim',  type=int,   default=512,  help='RNN_input_size (default: 300)')
	parser.add_argument('--RNN_emb_dim', type=int,   default=512,  help='RNN_hidden_size (default: 512)')	
	parser.add_argument('--Classifier_lr', type=float, default=1e-5,help='Classifier Learning Rate (default: 1e-5)')
	parser.add_argument('--Classifier_bs', type=int, default=18,help='batch_size for each Classifier iterations (default: 18)')
	parser.add_argument('--Classifier_Trans_dim', type=int,  default=4096, help='default: 4096')
	# parser.add_argument('--Classifier_ImgTrans_dim', type=int,  default=4096, help='default: 4096')
	# parser.add_argument('--Classifier_AnsTrans_dim', type=int,  default=4096, help='default: 4096')
	# parser.add_argument('--Classifier_QuesTrans_dim', type=int, default=4096, help='default: 4096')
	parser.add_argument('--Classifier_itr', type=int, default=5,help='Classifier Iteration (default: 18)')
	
	parser.add_argument('--ValueNN_RNN2Value_dim', type=int, default=500, help='ValueNN Embedding trans dimension  (default: 500)')
	parser.add_argument('--ValueNN_Img2Value_dim', type=int, default=500, help='ValueNN Images trans dimension (default: 500)')
	parser.add_argument('--ValueNN_itr', type=int, default=50, help='ValueNN training maximum Iteration  (default: 10)')
	parser.add_argument('--ValueNN_lr', type=float, default=1e-5,help='ValueNN Learning Rate (default: 18)')
	parser.add_argument('--ValueNN_bs', type=int, default=400,help='batch_size for each ValueNN iterations (default: 18)')

	parser.add_argument('--MC_RollOut_Size', type=int, default=3000, help='default: 4096')
	parser.add_argument('--MC_Root_size', type=int, default=5000,help='default: 1000')



	#Misc
	parser.add_argument('--ValueNNDataPath', type=str, default="../data/ValueNN/", help='path for storing ValueNN train data')
	parser.add_argument('--max_words_q', type=int, default=26, help='maximum words for a sentence(default: 26)')
	parser.add_argument('--global_itr', type=int, default=1000, help='maximum iteration(default: 1000)')
	parser.add_argument('--epoches', type=int, default=300, help='epoches(default: 300)')
	parser.add_argument('--seed', type=int, default=320, help='random seed (default: 320)')
	#GPU
	parser.add_argument('--cuda', action='store_true', default=True,help='enable CUDA training')
	return vars(parser.parse_args())

def Generate_MoteCarloTree_Root(Classifier, ValueNN, Imgs, Data, itr, MC_Root_size, **kwargs):
	print("\tGenerating MonteCarloTree Root")
	record = []
	cnt = 0
	lim = 0
	for index, img, ques, ans, targets in tqdm(Classifier_batch_generator(Imgs, Data, 100, 1)):
		_, states = Classifier.forward( ValueNN, 
										Variable(from_numpy(img) , volatile = True).cuda(), 
										Variable(from_numpy(ques), volatile = True).cuda(),
										Variable(from_numpy(ans) , volatile = True).cuda())
		for i in range(len(index)):
			step = np.random.randint(24,25)
			record.append([index[i], step, states[step][i, :]])
		cnt += len(index)
		if cnt >=lim:
			print(cnt,"/",MC_Root_size)
			lim += 1000
		if cnt >= MC_Root_size: break;

	return record

def Generate_ValueNN_train_data(Classifier, records, Img, Data, Itr, ValueNNDataPath, **kwargs):
	print ("\tGenerating ValueNN Train Data")
	Record = []
	for img_index, state, ques, ans, img, target in tqdm(fetch_records(records, Img, Data)):
		Tree = MonteCarloTree_Module(Classifier, state, ques, ans, img, target)
		value = Tree.Generate(100)
		Record.append([img_index, state.data.cpu().numpy(), value])
	file = open(path.join(ValueNNDataPath,"ValueNN_Train_" + str(Itr)),'wb')
	pickle.dump(Record, file)

def Train_ValueNN(ValueNN, img, global_itr, ValueNN_bs, ValueNNDataPath, ValueNN_lr, ValueNN_itr, **kwargs):
	imgs, states, targets = load_ValueNN_data(img, ValueNNDataPath, global_itr)
	optimizer = optim.Adam([{"params":ValueNN.parameters()}], lr =ValueNN_lr)
	print ("\tTraining ValueNN")
	for itr in tqdm(range(ValueNN_itr)):
		Losses = []
		for img, state, target in ValueNN_batch_generator(imgs, states, targets, ValueNN_bs):
			optimizer.zero_grad()
			value = ValueNN.forward(Variable(from_numpy(state)).cuda(), Variable(from_numpy(img)).cuda())
			loss = F.mse_loss(value.cpu().view(-1), Variable(from_numpy(target)))
			loss.backward()
			optimizer.step()
			Losses.append(loss.data.numpy())
			# print(loss.data.numpy())
		if itr % 5 == 0:
			print ("\t\tValueNN Iteration %d : %.6f"%(itr, np.array(Losses).mean()))

def Train_Classifier(Classifier, ValueNN, imgs, data, Classifier_bs, Classifier_lr, Classifier_itr, **kwargs):
	print("\tTraining Classifier")
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, Classifier.parameters()), lr =Classifier_lr)
	for itr in range(Classifier_itr):
		Losses = []
		for _, img, ques, ans, target in tqdm(Classifier_batch_generator(imgs, data, Classifier_bs, 1)):
			optimizer.zero_grad()
			confidence, _ = Classifier.forward(ValueNN, Variable(from_numpy(img).cuda()), 
				Variable(from_numpy(ques).long()).cuda(), Variable(from_numpy(ans).long()).cuda())
			loss = F.binary_cross_entropy(confidence.cpu(), Variable(from_numpy(target).double()))
			loss.backward()
			optimizer.step()
			Losses.append(loss.data.numpy())
		if itr % 5 == 0:
			print ("\tClassifier Iteration %d : %.3f"%(itr, np.array(Losses).mean()))

def Valid(Classifier, ValueNN, Img, data, Classifier_bs, **kwargs):
	print ("\tValiding")
	res,record = {},{}
	for index, img, ques, ans, targets in tqdm(Classifier_batch_generator(Img, data, 200 , 3)):
		confidences, _ = Classifier.forward(ValueNN, 
											Variable(from_numpy(img) , volatile = True).cuda(), 
											Variable(from_numpy(ques), volatile = True).cuda(), 
											Variable(from_numpy(ans) , volatile = True).cuda())

		confidences = confidences.cpu().data.numpy()
		for id, confidence, target in zip(index, confidences, targets): 
			# print id, confidence, target
			if (id // 4 not in record) or confidence[0] > record[id // 4]:
				record[id // 4] = confidence[0]
				res[id // 4] = target[0]

	acc = sum(res.values()) / len(record)
	print ('\tAccuracy of test: %.4f'%(acc))
	return acc

def Train(global_itr, model_save, **args):
	train_img, train_data, test_img, test_data, emb_matrix = Get_Data(**args)
	num_train = train_data['question'].shape[0]

	ValueNN    = ValueNN_Module(**args).double().cuda()
	Classifier = Classifier_Module(emb_matrix, **args).cuda()
	best = 0
	for itr in range(global_itr):
		print("Iteration %d"%(itr))
		record = Generate_MoteCarloTree_Root(Classifier, ValueNN, train_img, train_data, itr, **args)
		Generate_ValueNN_train_data(Classifier, record, train_img, train_data, itr, **args)
		Train_ValueNN(ValueNN, train_img, itr, **args)
		Train_Classifier(Classifier, ValueNN, train_img, train_data, **args)
		acc = Valid(Classifier, ValueNN, test_img, test_data, **args)
		if acc > best:
			best = acc
			torch.save({
				'ValueNN_state_dict': ValueNN.state_dict(),
				'Classifier_state_dict': Classifier.state_dict(),
				'parameters': args
				}, model_save + "RL_" +str(best))


if __name__ == '__main__':
	# torch.multiprocessing.set_start_method("spawn")
	args = Arguement()
	# with torch.cuda.device():
	Train(**args)
