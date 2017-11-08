import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import from_numpy
from torch.autograd import Variable
from torch.nn import Module, Linear, GRUCell, Embedding
from random import uniform, randint
from numpy import zeros, array, arange
class ValueNN_Module(Module):
	def __init__(self, **args):
		super(ValueNN_Module, self).__init__()
		self.RNN2Value  = Linear(args['RNN_emb_dim'], args['ValueNN_RNN2Value_dim'])
		self.Img2Value  = Linear(args['image_dim'], args['ValueNN_Img2Value_dim'])
		self.Regression = Linear(args['ValueNN_RNN2Value_dim'] + args['ValueNN_Img2Value_dim'], 1)
	def forward(self, rnn_emb, Img):
		print rnn_emb.dtype
		print Img.dtype
		RValue = self.RNN2Value(rnn_emb)
		IValue = self.Img2Value(Img)
		return self.Regression(torch.concat((RValue,IValue),0))


class Classifier_Module(Module):
	def __init__(self, Embed_matrix, **args):
		super(Classifier_Module, self).__init__()
		self.WordTrans    = Linear(args['word_dim'],  args['Classifier_QuesTrans_dim'])
		self.ImgTrans     = Linear(args['image_dim'], args['Classifier_ImgTrans_dim'])
		self.AnsTrans     = Linear(args['image_dim'], args['Classifier_AnsTrans_dim'])
		self.Classify     = Linear(args['Classifier_AnsTrans_dim'], args['classify_dim'])
		self.input_dim   = args['RNN_input_dim']
		self.emb_dim  = args['RNN_emb_dim']
		self.Embed_lookup = Embedding(*Embed_matrix.shape)
		self.Embed_lookup.weight.data.copy_(from_numpy(Embed_matrix))
		self.Embed_lookup.weight.requires_grad = False
		self.Cell = GRUCell(input_size = args['RNN_input_dim'], hidden_size = args['RNN_emb_dim'])

	def forward(self, ValueNN, Img, Ques, Ans):
		Init_state = zeros((Ques.shape[0], self.emb_dim))
		hidden_state = [Init_state]
		ques = self.Embed_lookup(from_numpy(Ques))
		ans = self.Embed_lookup(from_numpy(Ans))
		ans = ans.mean(dim = 1) 
		for word in Ques:
			choice = []
			for region in xrange(49):
				confidence = ValueNN.forward(hidden_state[-1], Img[:,region,:])
				choice.append(confidence)
			choice = array(Choice).argmax(axis = 1)
			select_img = img[arange(length), choice, :]
			word_ = self.WordTrans(word)
			img_  = self.ImgTrans(select_img)
			QI = torch.mul(word_ , img_)
			state = self.Cell(QI, hidden_state[-1])
			hidden_state.appned(state)
		return self.inference(ans, hidden_state), hidden_state

	def Inference(self, Ans, hidden_state):
		ans_ = self.AnsTrans(ans)
		QIA = torch.mul(hidden_state[-1], ans_)
		confidence = F.Softmax(self.Classify(QIA), axis = 1)
		return confidence

class MonteCarloTree_Module():
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


	def __init__(self, state, Ques, Ans, Img, Traget, Classifier):
		self.root = Node(state)
		self.words = Ques
		self.Ans = Classifier.Embed_lookup(Ans).mean(dim = 1)
		self.Img = Img
		self.Target = Target
		self.Classifier = Classifier


	def Sample(self, epsilon):
		p = self.root
		path = []
		for word in self.Ques:
			prob = uniform(0,1)
			next = randint(0, 48) if uniform(0,1) < epsilon else p.Score().argmax()
			if p.son[next] == None:
				next_state = self.Classifier.Cell(p.state, self.Img[next])
				p.son[next] = Node(next_state)
			p = p.son[next]
			path.append(next)
		confidence = self.Classifier.inference(self.Ans, p.state)
 		reward = 1 if confidence.argmax() == self.Target.argmax() else 0 
		p = self.root
		for node in path:
			p.win[node] += reward
			p = p.s[node]

	def Generate(self, Roll_Out_size):
		for num in xrange(Roll_Out_size):
			self.Sample(Img, Target, ValueNN, 0.2)
		return F.Softmax(self.root.Score(), axis = 1)
