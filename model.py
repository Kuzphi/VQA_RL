from __future__ import print_function

import torch
import math
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch import from_numpy, arange, Tensor
from torch.autograd import Variable
from torch.nn import Module, Linear, GRUCell, Embedding
from random import uniform, randint

class ValueNN_Module(Module):
	def __init__(self, **args):
		super(ValueNN_Module, self).__init__()
		self.RNN2Value  = Linear(args['RNN_emb_dim'], args['ValueNN_RNN2Value_dim']).double()
		self.Img2Value  = Linear(args['image_dim'], args['ValueNN_Img2Value_dim']).double()
		self.Regression = Linear(args['ValueNN_RNN2Value_dim'] + args['ValueNN_Img2Value_dim'], 1).double()
	def forward(self, rnn_emb, Img):
		RValue = self.RNN2Value(rnn_emb)
		IValue = self.Img2Value(Img)
		return self.Regression(torch.cat((RValue,IValue), dim = 1))

class Classifier_Module(Module):
	def __init__(self, Embed_matrix, **args):
		super(Classifier_Module, self).__init__()
		self.WordTrans    = Linear(args['word_dim'],  args['Classifier_Trans_dim']).double()
		self.ImgTrans     = Linear(args['image_dim'], args['Classifier_Trans_dim']).double()
		self.AnsTrans     = Linear(args['word_dim'],  args['RNN_output_dim']).double()
		self.Classify     = Linear(args['RNN_output_dim'], args['classify_dim']).double()
		self.img_dim      = args['image_dim']
		self.input_dim    = args['RNN_input_dim']
		self.emb_dim      = args['RNN_emb_dim']
		self.Embed_lookup = Embedding(*Embed_matrix.shape).double()
		self.Embed_lookup.weight.data.copy_(from_numpy(Embed_matrix).double())
		self.Embed_lookup.weight.requires_grad = False
		self.Cell = GRUCell(input_size = args['Classifier_Trans_dim'], hidden_size = args['RNN_emb_dim']).double()


	def _process_one(self, ValueNN, state, img, choice, region):
		confidence = ValueNN.forward(state, img)
		choice[:, region] = confidence.view(-1)


	def forward(self, ValueNN, Img, Ques, Ans):
		bs, length = Ques.data.size()
		Init_state = Variable(torch.zeros((bs, self.emb_dim)).double()).cuda()
		hidden_state = [Init_state]
		ques = self.Embed_lookup(Ques)
		ans = self.Embed_lookup(Ans)
		ans = ans.mean(dim = 1) 

		for step in range(length):
			choice = torch.zeros(bs, 49).cuda().double()			
			for region in range(49):
				confidence = ValueNN.forward(Variable(hidden_state[-1].data).cuda(), Img[:,region,:])
				choice[:, region] = confidence.view(-1).data

			# process = []
			# for region in range(49):
			# 	this_state = Variable(hidden_state[-1].data).cuda()
			# 	p = torch.multiprocessing.Process(target= self._process_one, args=  (ValueNN, this_state, Img[:,region,:], choice, region) )
			# 	p.start()
			# 	process.append(p)

			# for p in process:
			# 	p.join()

			_, choice = choice.max(dim = 1)
			select_img = Tensor(bs, self.img_dim).double()
			for i in range(bs):
				select_img[i,:] = Img[i, choice[i],:].data
			# select_img = Img[torch.arange(0, bs).int(), choice.int(), :]
			state = self.OneStep(ques[:,step,:].cuda(), Variable(select_img).cuda(), hidden_state[-1].cuda())
			hidden_state.append(state)
		return self.Inference(ans, hidden_state), hidden_state

	def OneStep(self, ques, img, hidden):
		img_  = self.ImgTrans(img)
		word_ = self.WordTrans(ques)
		QI = torch.mul(word_ , img_)
		return self.Cell(QI, hidden)

	def Inference(self, Ans, hidden_state):
		ans_ = self.AnsTrans(Ans)
		QIA = torch.mul(hidden_state[-1], ans_)
		confidence = F.softmax(self.Classify(QIA), dim = 1)
		return confidence

class Node():
	def __init__(self, state):
		self.state = state
		self.Q = self.P = 0
		self.son = [None] * 49
		self.cnt = np.array([1.] * 49)
		self.win = np.array([0.] * 49)
	def Score(self):
		# Upper Confidence Bound
		t = self.cnt.sum()
		score = np.divide(self.win, self.cnt) + np.sqrt( 2 * math.log(t) / self.cnt)
		return score

class MonteCarloTree_Module():
	def __init__(self, Classifier, state, Ques, Ans, Img, Target):
		self.root  		= Node(state.view(1,-1))
		self.Ans   		= Classifier.Embed_lookup(Variable(from_numpy(Ans.reshape(1,-1) ).long()).cuda()).mean(dim = 1)
		self.words 		= Classifier.Embed_lookup(Variable(from_numpy(Ques).long()).cuda())
		self.Img    	= Img 
		self.Target 	= Target
		self.Classifier = Classifier

	def Sample(self, epsilon):
		p = self.root
		path = []
		for word in self.words:
			prob = uniform(0,1)
			next = randint(0, 48) if uniform(0,1) < epsilon else p.Score().argmax()
			if p.son[next] == None:
				# print p.state.shape, self.Img[:,next,:].shape
				this_img    = Variable(from_numpy(self.Img[next,:].reshape(1,-1))).cuda()
				next_state  = self.Classifier.OneStep(word.view(1,-1), this_img, p.state)
				p.son[next] = Node(next_state)
			p = p.son[next]
			path.append(next)
		confidence = self.Classifier.Inference(self.Ans, p.state)
		reward = 0
		if confidence.data.cpu().numpy().argmax() == self.Target.argmax():
			reward = 1
 		# reward = 1 if confidence.data.cpu().numpy().argmax() == self.Target.argmax() else 0 
		p = self.root
		for node in path:
			p.win[node] += reward
			p = p.son[node]

	def Generate(self, Roll_Out_size):
		print("\t\tGenerating Sample")
		for num in range(Roll_Out_size):
			self.Sample(0.2)
		x = self.root.win / self.root.cnt.sum()
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum()
