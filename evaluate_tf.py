import torch
import torch.nn as nn
from train import sample_batch
from data_loader import load_questions_answers, load_image_features

# disable GPU info output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

def main():

	print "Reading fc7 features"
	fc7_features, image_id_list = data_loader.load_fc7_features(args.data_dir, 'val')
	print "FC7 features", fc7_features.shape
	print "image_id_list", image_id_list.shape

	image_id_map = {}
	for i in xrange(len(image_id_list)):
		image_id_map[ image_id_list[i] ] = i

	ans_map = { qa_data['answer_vocab'][ans] : ans for ans in qa_data['answer_vocab']}

	model_options = {
		'num_lstm_layers' : args.num_lstm_layers,
		'rnn_size' : args.rnn_size,
		'embedding_size' : args.embedding_size,
		'word_emb_dropout' : args.word_emb_dropout,
		'image_dropout' : args.image_dropout,
		'fc7_feature_length' : args.fc7_feature_length,
		'lstm_steps' : qa_data['max_question_length'] + 1,
		'q_vocab_size' : len(qa_data['question_vocab']),
		'ans_vocab_size' : len(qa_data['answer_vocab'])
	}
	
	
	
	model = vis_lstm_model.Vis_lstm_model(model_options)
	input_tensors, t_prediction, t_ans_probab = model.build_generator()
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()

	avg_accuracy = 0.0
	avg_acc_top3 = 0.0
	total = 0
	saver.restore(sess, args.model_path)
	
	batch_no = 0
	total_batches = len(qa_data['validation']) // args.batch_size + 1
	while (batch_no*args.batch_size) < len(qa_data['validation']):
		sentence, answer, fc7 = get_batch(batch_no, args.batch_size, 
			fc7_features, image_id_map, qa_data, 'val')
		
		pred, ans_prob = sess.run([t_prediction, t_ans_probab], feed_dict={
						input_tensors['fc7']:fc7,
						input_tensors['sentence']:sentence,
				})
		
		batch_no += 1
		if args.debug:
			for idx, p in enumerate(pred):
				print ans_map[p], ans_map[ np.argmax(answer[idx])]

		correct_predictions = np.equal(pred, np.argmax(answer, 1))
		correct_predictions = correct_predictions.astype('float32')
		accuracy = correct_predictions.mean()
		print "[{}/{}] Acc = {}".format(batch_no, total_batches, accuracy)
		avg_accuracy += accuracy

		# correct_prediction for top 3 answers
		ans_prob_top_idx = np.argsort(-ans_prob, axis=1)
		ans_idx = np.argmax(answer, 1)
		top3_correct = (ans_prob_top_idx == ans_idx[:, np.newaxis])[:, :3]
		top3_correct = np.sum(top3_correct, axis=1) > 0
		top3_acc = top3_correct.mean()
		print "[{}/{}] Top 3 acc = {}".format(batch_no, total_batches, top3_acc)
		avg_acc_top3 += top3_acc

		total += 1
	
	print "Acc", avg_accuracy/total
	print "Top 3 acc", avg_acc_top3/total


def get_batch(batch_no, batch_size, fc7_features, image_id_map, qa_data, split):
	qa = None
	if split == 'train':
		qa = qa_data['training']
	else:
		qa = qa_data['validation']

	si = (batch_no * batch_size)%len(qa)
	ei = min(len(qa), si + batch_size)
	n = ei - si
	sentence = np.ndarray( (n, qa_data['max_question_length']), dtype = 'int32')
	answer = np.zeros( (n, len(qa_data['answer_vocab'])))
	fc7 = np.ndarray( (n,4096) )

	count = 0

	for i in range(si, ei):
		sentence[count,:] = qa[i]['question'][:]
		answer[count, qa[i]['answer']] = 1.0
		fc7_index = image_id_map[ qa[i]['image_id'] ]
		fc7[count,:] = fc7_features[fc7_index][:]
		count += 1
	
	return sentence, answer, fc7

if __name__ == '__main__':
	main()
