import json
import argparse
from os.path import isfile, join
import re
import numpy as np
import pprint
import pickle
import collections
import pdb

def prepare_training_data(token_type='word', version = 2, data_dir = 'data', num_que=20000):
        """Generate training and validation data from json file, as well as the question_vocab, answer_vocab and max_question_size
        
        Keyword Arguments:
                version {int} -- which VQA dataset (default: {2})
                data_dir {str} -- data directory (default: {'data'})
        
        Returns:
                dict -- data dictionary
        """
        if version == 1:
                t_q_json_file = join(data_dir, 'vqa/MultipleChoice_mscoco_train2014_questions.json')
                t_a_json_file = join(data_dir, 'vqa/mscoco_train2014_annotations.json')

                v_q_json_file = join(data_dir, 'vqa/MultipleChoice_mscoco_val2014_questions.json')
                v_a_json_file = join(data_dir, 'vqa/mscoco_val2014_annotations.json')
                if token_type == 'word':
                        qa_data_file = join(data_dir, 'vqa/qa_data_file1.pkl')
                        vocab_file = join(data_dir, 'vqa/vocab_file1.pkl')
                elif token_type == 'bigram':
                        qa_data_file = join(data_dir, 'vqa/bi_qa_data_file1.pkl')
                        vocab_file = join(data_dir, 'vqa/bi_vocab_file1.pkl')
        else:
                t_q_json_file = join(data_dir, 'vqa/v2_OpenEnded_mscoco_train2014_questions.json')
                t_a_json_file = join(data_dir, 'vqa/v2_mscoco_train2014_annotations.json')

                v_q_json_file = join(data_dir, 'vqa/v2_OpenEnded_mscoco_val2014_questions.json')
                v_a_json_file = join(data_dir, 'vqa/v2_mscoco_val2014_annotations.json')
                if token_type == 'word':
                        qa_data_file = join(data_dir, 'qa_data_file2_'+str(num_que)+'.pkl')
                        vocab_file = join(data_dir, 'vocab_file2_'+str(num_que)+'.pkl')
                elif token_type == 'bigram':
                        qa_data_file = join(data_dir, 'bi_qa_data_file2_'+str(num_que)+'.pkl')
                        vocab_file = join(data_dir, 'bi_vocab_file2_'+str(num_que)+'.pkl')

        # IF ALREADY EXTRACTED
        # qa_data_file = join(data_dir, 'qa_data_file{}.pkl'.format(version))
        # if isfile(qa_data_file):
        #       with open(qa_data_file) as f:
        #               data = pickle.load(f)
        #               return data

        print("Loading Training questions")
        with open(t_q_json_file) as f:
                t_questions = json.loads(f.read())
        
        print("Loading Training anwers")
        with open(t_a_json_file) as f:
                t_answers = json.loads(f.read())

        print("Loading Val questions")
        with open(v_q_json_file) as f:
                v_questions = json.loads(f.read())
        
        print("Loading Val answers")
        with open(v_a_json_file) as f:
                v_answers = json.loads(f.read())

        
        print("Ans", len(t_answers['annotations']), len(v_answers['annotations']))
        print("Qu", len(t_questions['questions']), len(v_questions['questions']))

        answers = t_answers['annotations'][:num_que//2] + v_answers['annotations'][:num_que//2]
        questions = t_questions['questions'][:num_que//2] + v_questions['questions'][:num_que//2]
        
        # find the top 3000 answers and their frequencies
        answer_vocab = make_answer_vocab(answers) 
        # find soft version of answers
        soft_answers = make_soft_answers(answer_vocab, answers)
        # find the most frequent words in questions and their frequencies, as well as the max_question_length
        question_vocab, max_question_length = make_questions_vocab(token_type, questions, answers, answer_vocab) 
        print("Question Vocabulary Size", len(question_vocab))
        print("Max Question Length", max_question_length)
        # only need words
        word_regex = re.compile(r'\w+')
        training_data = []
        for i,question in enumerate( t_questions['questions'][:num_que//2]):
                ans = t_answers['annotations'][i]['multiple_choice_answer']
                # only need questions that has the top 3000 answers
                if ans in answer_vocab:
                        training_data.append({
                                'image_id' : t_answers['annotations'][i]['image_id'],
                                'question' : np.zeros(max_question_length),
                                'answer' : answer_vocab[ans],
                                'answers': soft_answers[t_answers['annotations'][i]['question_id']]
                                })
                        question_words = re.findall(word_regex, question['question'])
                        if token_type == 'bigram':
                                question_words = [question_words[i]+' '+question_words[i+1] for i,x in enumerate(question_words[:-1])]
                        base = max_question_length - len(question_words)
                        for i in range(0, len(question_words)):
                                training_data[-1]['question'][base + i] = question_vocab[ question_words[i] ] if question_words[i] in question_vocab else question_vocab['UNK']

        print("Training Data", len(training_data))
        val_data = []
        for i,question in enumerate( v_questions['questions'][:num_que//2]):
                ans = v_answers['annotations'][i]['multiple_choice_answer']
                if ans in answer_vocab:
                        val_data.append({
                                'image_id' : v_answers['annotations'][i]['image_id'],
                                'question' : np.zeros(max_question_length),
                                'answer' : answer_vocab[ans],
                                'answers': soft_answers[v_answers['annotations'][i]['question_id']]
                                })
                        question_words = re.findall(word_regex, question['question'])
                        if token_type == 'bigram':
                                question_words = [question_words[i]+' '+question_words[i+1] for i,x in enumerate(question_words[:-1])]
                        base = max_question_length - len(question_words)
                        for i in range(0, len(question_words)):
                                val_data[-1]['question'][base + i] = question_vocab[ question_words[i] ] if question_words[i] in question_vocab else question_vocab['UNK']

        print("Validation Data", len(val_data))
        # save all data in the `qa_data.pkl` file
        data = {
                'training' : training_data,
                'validation' : val_data,
                'answer_vocab' : answer_vocab,
                'question_vocab' : question_vocab,
                'max_question_length' : max_question_length
        }

        print("Saving qa_data")
        with open(qa_data_file, 'wb') as f:
                pickle.dump(data, f)

        # save all data in the `vocab_file{version}.pkl` file
        with open(vocab_file, 'wb') as f:
                vocab_data = {
                        'answer_vocab' : data['answer_vocab'],
                        'question_vocab' : data['question_vocab'],
                        'max_question_length' : data['max_question_length']
                }
                pickle.dump(vocab_data, f)

        return data
        
def load_questions_answers(token_type='word', version = 2, data_dir = 'data'):
        if token_type == 'word':
                qa_data_file = join(data_dir, 'qa_data_file{}.pkl'.format(version))
        elif token_type == 'bigram':
                qa_data_file = join(data_dir, 'bi_qa_data_file{}.pkl'.format(version))

        if isfile(qa_data_file):
                with open(qa_data_file, 'rb') as f:
                        data = pickle.load(f)
                        return data

def load_questions_answers_small(token_type='word', version = 2, data_dir = 'data', num_que=20000):
        if token_type == 'word':
                qa_data_file = join(data_dir, 'qa_data_file{}_{}.pkl'.format(version, num_que))
        elif token_type == 'bigram':
                qa_data_file = join(data_dir, 'bi_qa_data_file{}_{}.pkl'.format(version, num_que))

        if isfile(qa_data_file):
                with open(qa_data_file, 'rb') as f:
                        data = pickle.load(f)
                        return data

def get_question_answer_vocab(token_type='word', version = 2, data_dir = 'data'):
        if token_type == 'word':
                vocab_file = join(data_dir, 'vocab_file{}.pkl'.format(version))
        elif token_type == 'bigram':
                vocab_file = join(data_dir, 'bi_vocab_file{}.pkl'.format(version))
        
        vocab_data = pickle.load(open(vocab_file, 'rb'))
        return vocab_data

def make_answer_vocab(answers):
        top_n = 1000
        answer_frequency = {} 
        for annotation in answers:
                answer = annotation['multiple_choice_answer']
                if answer in answer_frequency:
                        answer_frequency[answer] += 1
                else:
                        answer_frequency[answer] = 1

        answer_frequency_tuples = [ (-frequency, answer) for answer, frequency in answer_frequency.items()]
        answer_frequency_tuples.sort()
        # pdb.set_trace()
        answer_frequency_tuples = answer_frequency_tuples[0:top_n-1]

        answer_vocab = {}
        for i, ans_freq in enumerate(answer_frequency_tuples):
                # print i, ans_freq
                ans = ans_freq[1]
                answer_vocab[ans] = i

        answer_vocab['UNK'] = top_n - 1
        return answer_vocab

def make_soft_answers(answer_vocab, answers):
        ans_dict = dict()
        for a in answers:
                ans = a['answers']
                q_id = a['question_id']
                ans_dict[q_id] = dict()
                a_list = list()
                for an in ans:
                        if an['answer'] in answer_vocab:
                                a_list.append(an['answer'])

                count = collections.Counter(a_list)
                for w, v in count.items():
                        ans_dict[q_id][answer_vocab[w] ] = v / float(len(a_list))

        return ans_dict

def make_questions_vocab(token_type, questions, answers, answer_vocab):
        word_regex = re.compile(r'\w+')
        question_frequency = {}

        max_question_length = 0
        for i,question in enumerate(questions):
                # answer for the question
                ans = answers[i]['multiple_choice_answer']
                count = 0
                # answer is among top 3000
                if ans in answer_vocab:
                        # tokenization
                        question_words = re.findall(word_regex, question['question'])
                        if token_type == 'bigram':
                                question_words = [question_words[i]+' '+question_words[i+1] for i,x in enumerate(question_words[:-1])]
                        # update frequency for each token
                        for qw in question_words:
                                if qw in question_frequency:
                                        question_frequency[qw] += 1
                                else:
                                        question_frequency[qw] = 1
                                count += 1
                if count > max_question_length:
                        max_question_length = count

        # pdb.set_trace()
        # set up the minimum frequency
        qw_freq_threhold = 0
        qw_tuples = [ (-frequency, qw) for qw, frequency in question_frequency.items()]
        # qw_tuples.sort()

        qw_vocab = {}
        for i, qw_freq in enumerate(qw_tuples):
                frequency = -qw_freq[0]
                qw = qw_freq[1]
                # print frequency, qw
                if frequency > qw_freq_threhold:
                        # +1 for accounting the zero padding for batc training
                        qw_vocab[qw] = i + 1
                else:
                        break

        qw_vocab['UNK'] = len(qw_vocab) + 1

        return qw_vocab, max_question_length


def load_image_features(data_dir, split):
    import h5py
    features = None
    image_id_list = None
    with h5py.File( join( data_dir, (split + '.h5')),'r') as hf:
        features = np.array(hf.get('features'))
    with h5py.File( join( data_dir, (split + '_image_id_list.h5')),'r') as hf:
        image_id_list = np.array(hf.get('image_id_list'))
    return features, image_id_list

def load_image_features_small(data_dir, split, num_que=20000):
    import h5py
    features = None
    image_id_list = None
    with h5py.File( join( data_dir, (split + '_'+str(num_que)+'.h5')),'r') as hf:
        features = np.array(hf.get('features'))
    with h5py.File( join( data_dir, (split + '_image_id_list_'+str(num_que)+'.h5')),'r') as hf:
        image_id_list = np.array(hf.get('image_id_list'))
    return features, image_id_list

def load_image_features_4096(data_dir, split):
    import h5py
    features = None
    image_id_list = None
    with h5py.File( join( data_dir, (split + '_4096.h5')),'r') as hf:
        features = np.array(hf.get('features'))
    with h5py.File( join( data_dir, (split + '_image_id_list_4096.h5')),'r') as hf:
        image_id_list = np.array(hf.get('image_id_list'))
    return features, image_id_list
