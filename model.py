import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import from_numpy, arange, tensor
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
		self.input_dim    = args['RNN_input_dim']
		self.emb_dim      = args['RNN_emb_dim']
		self.Embed_lookup = Embedding(*Embed_matrix.shape).double()
		self.Embed_lookup.weight.data.copy_(from_numpy(Embed_matrix).double())
		self.Embed_lookup.weight.requires_grad = False
		self.Cell = GRUCell(input_size = args['Classifier_Trans_dim'], hidden_size = args['RNN_emb_dim']).double()

	def forward(self, ValueNN, Img, Ques, Ans):
		bs, length = Ques.data.size()
		Init_state = Variable(torch.zeros((bs, self.emb_dim)).double())
		hidden_state = [Init_state]
		ques = self.Embed_lookup(Ques)
		ans = self.Embed_lookup(Ans)
		ans = ans.mean(dim = 1) 
		for step in xrange(length):
			choice = torch.zeros(4, 49).double()
			for region in xrange(49):
				confidence = ValueNN.forward(Variable(hidden_state[-1].data, requires_grad = False), Img[:,region,:]).data
				choice[:, region] = confidence.view(-1)
			_, choice = choice.max(dim = 1)
			select_img = Img[torch.arange(0, bs).long(), choice.long(), :]
			word_ = self.WordTrans(ques[:,step,:])
			img_  = self.ImgTrans(select_img)
			QI = torch.mul(word_ , img_)
			state = self.Cell(QI, hidden_state[-1])
			hidden_state.append(state)
		return self.Inference(ans, hidden_state), hidden_state

	def Inference(self, Ans, hidden_state):
		ans_ = self.AnsTrans(Ans)
		# print hidden_state[-1], ans_
		QIA = torch.mul(hidden_state[-1], ans_)
		confidence = F.softmax(self.Classify(QIA), dim = 1)
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
