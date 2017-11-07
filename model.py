import argparse
import torch
from torch.nn import Linear, GRUCell
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from load import Get_Data
from random import uniform, randint
class ValueNN(nn.model):
	def __init__(self, **kwargs):
		super(ValueNN, self).__init__()
		self.Rnn2Value  = Linear(rnn_emb_size, RNN2Value)
		self.Img2Value  = Linear(image_dim, Img2Value)
		self.Regression = Linear(Img2Value + RNN2Value, 1)
	def forward(self, rnn_emb, Img):
		RValue = self.RNN2Value(rnn_emb)
		IValue = self.Img2Value(Img)
		return self.Regression(torch.concat((RValue,IValue),0))


class Classfier(nn.model):

	def __init__(self, **kwargs):
		super(Classfier, self).__init()__
		self.WordTrans    = Linear(word_dim,  Classfier_QuesTrans_dim)
		self.ImgTrans     = Linear(image_dim, Classfier_ImgTrans_dim)
		self.AnsTrans     = Linear(image_dim, Classfier_AnsTrans_dim)
		self.Classify     = Linear(Classfier_AnsTrans_dim, 2)
		self.input_size   = RNN_input_size
		self.hidden_size  = RNN_hidden_size
		self.Embed_lookup = Embedding(len(Embed_matrix), 300)
		self.Embed_lookup.weight.data.copy_(Embed_matrix)
		self.Embed_lookup.weight.requires_grad = False
		self.Cell = nn.GRUCell(input_size = RNN_input_size, hidden_size = RNN_hidden_size)

	def forward(self, Ques, Ans, ValueNN, Embed_matrix, Img):
		length = Ques.size()[0]
		hidden_state = [np.zeros(Ques,shape[0], self.hidden_size)]
		ques = self.Embed_lookup(Ques)
		ans = self.Embed_lookup(ans)
		ans = ans.mean(ans, dim = 1) 
		for word in Ques:
			choice = []
			for region in xrange(49):
				confidence = ValueNN.forward(hidden_state[-1], Img[:,region,:])
				choice.append(confidence)
			choice = np.array(Choice).argmax(axis = 1)
			select_img = img[np.arange(length), choice, :]
			word_ = self.WordTrans(word)
			img_  = self.ImgTrans(select_img)
			QI = torch.mul(word_ , img_)
			state = self.Cell(QI, hidden_state[-1])
			hidden_state.appned(state)
		return inference(ans, hidden_state), hidden_state

	def Inference(Ans, hidden_state):
		ans_ = self.AnsTrans(ans)
		QIA = torch.mul(hidden_state[-1], ans_)
		confidence = F.Softmax(self.Classify(QIA), axis = 1)
		return confidence

class MonteCarloTree():
	class Node():
		def __init__(self, state):
			self.state = state
			self.Q = self.P = 0
			self.s = [None] * 49
			self.cnt = np.array([1.] * 49)
			self.win = np.array([0.] * 49)
		def Score():
			# Upper Confidence Bound
			t = self.cnt.sum()
			score = self.win / self.cnt + sqrt( 2 * log(t) / self.cnt)
			return score


	def __init__(self, state, Ques, Ans, Img):
		self.root = Node(state)
		self.words = Ques
		self.Ans = Ans
		self.Img = Img
		self.Target = Target


	def Sample(self, Classfier, epsilon):
		p = self.root
		path = []
		for word in self.Ques:
			prob = uniform(0,1)
			next = randint(0, 48) if uniform(0,1) < epsilon else p.Score().argmax()
			if p.son[next] == None:
				next_state = Classfier.Cell(p.state, self.Img[next])
				p.son[next] = Node(next_state)
			p = p.son[next]
			path.append(next)
		confidence = Classfier.inference(self.Ans, p.state)
 		reward = 1 if confidence.argmax() == Target.argmax() else 0 
		p = self.root
		for node in path:
			p.win[node] += reward
			p = p.s[node]

	def Generate(self, Sample_size, Classfier):
		for num in xrange(Sample_size):
			self.Sample(Img, Target, Classfier, ValueNN, 0.2)
		return self.root.state, F.Softmax(self.root.Score(), axis = 1)














