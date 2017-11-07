

def right_align(seq, lengths):
	v = np.zeros(np.shape(seq))
	N = np.shape(seq)[1]
	for i in range(np.shape(seq)[0]):
		v[i][N-lengths[i]:N]=seq[i][0:lengths[i]]
	return v

def Load_data(input_img_h5, input_ques_h5):
	data = {}
	print('loading image feature...')
	with h5py.File(input_img_h5_train,'r') as hf:
	# -----0~82459------  at most 47000
		tem = hf.get('images')
		img_feature = np.array(tem).reshape(-1, 49, 2048)
	print('loading h5 file...')
	with h5py.File(input_ques_h5,'r') as hf:
		# total number of training data is 215375
		# question is (26, )
		tem = hf.get('ques')
		train_data['question'] = np.array(tem)-1
		# max length is 23
		tem = hf.get('ques_length')
		train_data['length_q'] = np.array(tem)
		# total 82460 img
		tem = hf.get('img_pos')
		# convert into 0~82459
		train_data['img_list'] = np.array(tem)-1
		# answer
		tem = hf.get('ans')
		train_data['answer'] = np.array(tem)-1

		tem = hf.get('ans_length')
		train_data['length_a'] = np.array(tem)

		tem = hf.get('target')
		train_data['target'] = np.transpose(np.vstack((np.array(tem), 1-np.array(tem))))

		train_data['emb_matrix'] = np.array(hf.get('emb_matrix'))

		train_data['ques_pos'] = np.array(hf.get('pos_train_ques')) - 1
		train_data['ans_pos'] = np.array(hf.get('pos_train_ans')) - 1
	print('question & answer aligning')
	data['question'] = right_align(train_data['question'], train_data['length_q'])
	data['answer']   = right_align(train_data['answer'], train_data['length_a'])
	data['ques_pos'] = right_align(train_data['ques_pos'], train_data['length_q'])
	data['ans_pos']  = right_align(train_data['ans_pos'], train_data['length_a'])
	return img_feature, data

def Get_Data(**kwargs):
	with h5py.File(input_ques_h5,'r') as hf:
		emb_matrix = np.array(hf.get('emb_matrix'))
	train_img_feature, train_data = Load_data(input_img_h5_train, input_ques_h5)
	test_img_feature, test_data = Load_data(input_img_h5_test, input_ques_h5)

	return train_img_feature, train_data, test_img_feature, test_data, emb_matrix
