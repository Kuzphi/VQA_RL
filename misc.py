import numpy as np
from numpy import array
import h5py
import pickle
import itertools
def right_align(seq, lengths):
	v = np.zeros(np.shape(seq), dtype = np.int64)
	N = np.shape(seq)[1]
	for i in range(np.shape(seq)[0]):
		v[i][N-lengths[i]:N]=seq[i][0:lengths[i]]
	return v

def Load_data(img_h5, ques_h5, data_type, testing = 1):
	data = {}
	print('loading' + data_type + ' image feature...')
	# -----0~82459------  at most 47000
	if testing:
		if data_type == '_train':
			img_feature = np.zeros((20044, 49, 2048))
		else:
			img_feature = np.zeros((8609, 49, 2048))
	else:
		with h5py.File(img_h5,'r') as hf:
			tem = hf.get('images' + data_type)
			img_feature = np.array(tem).reshape(-1, 49, 2048)
			print img_feature.shape

	print('loading' + data_type + ' h5 file...')
	with h5py.File(ques_h5,'r') as hf:
		# total number of training data is 215375
		# question is (26, )
		tem = hf.get('ques' + data_type)
		data['question'] = np.array(tem, dtype = np.int64) - 1

		# max length is 23
		tem = hf.get('ques_length' + data_type)
		data['length_q'] = np.array(tem)

		# 82460 img convert into 0~82459
		tem = hf.get('img_pos' + data_type)
		data['img_list'] = np.array(tem)-1
		# answer
		tem = hf.get('ans' + data_type)
		data['answer'] = np.array(tem, dtype = np.int64) - 1

		tem = hf.get('ans_length' + data_type)
		data['length_a'] = np.array(tem)

		tem = hf.get('target' + data_type)
		data['target'] = np.transpose(np.vstack((np.array(tem, dtype = np.int32), 1-np.array(tem, dtype = np.int32))))

		data['ques_pos'] = np.array(hf.get('pos' + data_type +'_ques')) - 1
		data['ans_pos']  = np.array(hf.get('pos' + data_type +'_ans'))  - 1
	print('question & answer aligning')
	data['question'] = right_align(data['question'], data['length_q'])
	data['answer']   = right_align(data['answer'],   data['length_a'])
	data['ques_pos'] = right_align(data['ques_pos'], data['length_q'])
	data['ans_pos']  = right_align(data['ans_pos'],  data['length_a'])
	return img_feature, data

def Get_Data(ques_h5, train_img_h5, test_img_h5, **kwargs):
	with h5py.File(ques_h5,'r') as hf:
		emb_matrix = np.array(hf.get('emb_matrix'))
	train_img_feature, train_data = Load_data(train_img_h5, ques_h5, "_train")
	test_img_feature, test_data = Load_data(test_img_h5, ques_h5, "_test")

	return train_img_feature, train_data, test_img_feature, test_data, emb_matrix

def Classifier_batch_generator(Imgs, data, batch_size):
	sz = data['question'].shape[0]
	index = np.arange(sz / 4) * 4
	np.random.shuffle(index)
	i = 0
	while i < index.shape[0]:
		pos = index[i:i + batch_size]
		neg = [np.random.choice([id+1, id+2, id+3], 3, replace = False) for id in pos]
		neg = list(itertools.chain(*neg))
		np.random.shuffle(neg)
		batch_index = list(pos) + list(neg)
		np.random.shuffle(batch_index)

		question = data['question'][batch_index,:]
		length_q = data['length_q'][batch_index]
		answer   = data['answer'][batch_index]
		length_a = data['length_a'][batch_index]
		img_list = data['img_list'][batch_index]
		target   = data['target'][batch_index]

		ques_pos = data['ques_pos'][batch_index,:]
		ans_pos = data['ans_pos'][batch_index,:]

		target = np.array(target);
		img = Imgs[img_list,:]
		yield batch_index, img, question, answer, length_a, target

def ValueNN_batch_generator(img, state, targets, batch_size):
	idx = np.random.shuffle(np.arange(img.shape[0]))
	while i < img.shape[0]:
		index = idx[i:i + batch_size]
		yield img[index], state[index], targets[index]

def load_ValueNN_data(global_img, Path, itr):
	with open(path.join(ValueNNDataPath,"ValueNN Train " + str(itr)), 'rb') as file:
		record = pickle.load(file)
	img, states, target = [], [], []
	for index, state, target in record:
		states = states + [state] * 49
		targets = targets + target
		img = img + list(global_img[index])
	return array(img), array(states), array(targets)


