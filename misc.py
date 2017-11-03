import numpy as np
from numpy import array
def right_align(seq, lengths):
	v = np.zeros(np.shape(seq))
	N = np.shape(seq)[1]
	for i in range(np.shape(seq)[0]):
		v[i][N-lengths[i]:N]=seq[i][0:lengths[i]]
	return v

def Load_data(input_img_h5, input_ques_h5, data_type):
	data = {}
	print('loading' + data_type + ' image feature...')
	with h5py.File(input_img_h5,'r') as hf:
	# -----0~82459------  at most 47000
		tem = hf.get('images')
		img_feature = np.array(tem).reshape(-1, 49, 2048)
		
	print('loading' + data_type + ' h5 file...')
	with h5py.File(input_ques_h5,'r') as hf:
		# total number of training data is 215375
		# question is (26, )
		tem = hf.get('ques' + data_type)
		data['question'] = np.array(tem)-1

		# max length is 23
		tem = hf.get('ques_length' + data_type)
		data['length_q'] = np.array(tem)

		# 82460 img convert into 0~82459
		tem = hf.get('img_pos' + data_type)
		data['img_list'] = np.array(tem)-1
		# answer
		tem = hf.get('ans' + data_type)
		data['answer'] = np.array(tem)-1

		tem = hf.get('ans_length' + data_type)
		data['length_a'] = np.array(tem)

		tem = hf.get('target' + data_type)
		data['target'] = np.transpose(np.vstack((np.array(tem), 1-np.array(tem))))

		data['ques_pos'] = np.array(hf.get('pos' + data_type +'_ques')) - 1
		data['ans_pos']  = np.array(hf.get('pos' + data_type +'_ans'))  - 1
	print('question & answer aligning')
	data['question'] = right_align(data['question'], data['length_q'])
	data['answer']   = right_align(data['answer'],   data['length_a'])
	data['ques_pos'] = right_align(data['ques_pos'], data['length_q'])
	data['ans_pos']  = right_align(data['ans_pos'],  data['length_a'])
	return img_feature, data

def Get_Data(**kwargs):
	with h5py.File(input_ques_h5,'r') as hf:
		emb_matrix = np.array(hf.get('emb_matrix'))
	train_img_feature, train_data = Load_data(input_img_h5_train, input_ques_h5, "_train")
	test_img_feature, test_data = Load_data(input_img_h5_test, input_ques_h5, "_test")

	return train_img_feature, train_data, test_img_feature, test_data, emb_matrix

def batch_generator(img, data, batch_size):
	sz = data['question'].shape[0]
	index = np.random.shuffle(np.array(range(sz / 4)) * 4)
	i = 0
	while i < index.shape(0):
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

		ques_pos = train_data['ques_pos'][batch_index,:]
		ans_pos = train_data['ans_pos'][batch_index,:]

		target = np.array(current_target);
		img = img_feature_train[img_list,:]
		yield img, question, answer, length_a
 


