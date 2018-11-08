import json
import re
import numpy as np
import pickle
import collections
import torch
import os
from os.path import join
import pdb


def clean_state_dict(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k[:min(6,len(k))] == 'module' else k # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def sample_batch_hard(batch_no, batch_size, features, image_id_map, qa, qa_data):
    si = (batch_no * batch_size)%len(qa)
    ei = min(len(qa), si + batch_size)
    n = ei - si
    sentence = np.ndarray( (n, qa_data['max_question_length']), dtype=int) # [N, 22]
    answers = np.zeros((n,), dtype=int) # [N,]
    features = torch.empty( (n,49,1024) ) # [N, 49, 1024]
    count = 0
    for i in range(si, ei):
        sentence[count,:] = qa[i]['question'][:]
        answers[count] = qa[i]['answer']
        features_index = image_id_map[ qa[i]['image_id'] ]
        features[count,:,:] = features[features_index, :, :]
        count += 1
    return features, torch.tensor(sentence), torch.tensor(answers)

def sample_batch_soft(batch_no, batch_size, features, image_id_map, qa, qa_data):
    si = (batch_no * batch_size)%len(qa)
    ei = min(len(qa), si + batch_size)
    n = ei - si
    sentence = np.ndarray( (n, qa_data['max_question_length']), dtype=int) # [N, 22]
    soft_answers = np.zeros((n, len(qa_data['answer_vocab'])), dtype=int) # [N, answer_vocab_size]
    answers = np.zeros((n,), dtype=int) # [N,]
    features = torch.empty( (n,49,1024) ) # [N, 49, 1024]

    count = 0
    for i in range(si, ei):
        sentence[count,:] = qa[i]['question'][:]
        sparse_soft_answers = qa[i]['answers']
        idx = list(sparse_soft_answers.keys())
        probs = list(sparse_soft_answers.values())
        soft_answers[count,idx] = probs
        answers[count] = qa[i]['answer']
        features_index = image_id_map[ qa[i]['image_id'] ]
        features[count,:,:] = features[features_index, :, :]
        count += 1
    return features, torch.tensor(sentence), torch.tensor(soft_answers), torch.tensor(answers)


contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
               "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
               "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
               "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
               "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
               "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
               "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
               "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
               "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
               "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
               "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
               "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
               "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
               "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
               "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
               "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
               "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
               "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
               "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
               "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
               "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
               "youll": "you'll", "youre": "you're", "youve": "you've"}
manualMap    = {'none': '0',
                'zero': '0',
                'one': '1',
                'two': '2',
                'three': '3',
                'four': '4',
                'five': '5',
                'six': '6',
                'seven': '7',
                'eight': '8',
                'nine': '9',
                'ten': '10'
              }

def contract_word(sent):
    """
    input: sentence, word list
    output: sentence of lower case of words with the same length as input after contraction and number replacement.
    """
    sent_processed = list()
    for w in sent:
        w = w.lower()
        if w in contractions:
            w = contractions[w]
        if w in manualMap:
            w = manualMap[w]
        sent_processed.append(w)
    return sent_processed

def prepare_training_data(data_dir = 'data', version=2, num_ans=1000, answer_type='all'):

    assert answer_type in ('all', 'other', 'yes/no', 'number'), 'answer_type is not satisfied'

    if version == 1:
        t_q_json_file = join(data_dir, 'vqa/MultipleChoice_mscoco_train2014_questions.json')
        t_a_json_file = join(data_dir, 'vqa/mscoco_train2014_annotations.json')

        v_q_json_file = join(data_dir, 'vqa/MultipleChoice_mscoco_val2014_questions.json')
        v_a_json_file = join(data_dir, 'vqa/mscoco_val2014_annotations.json')
    else:
        t_q_json_file = join(data_dir, 'vqa/v2_OpenEnded_mscoco_train2014_questions.json')
        t_a_json_file = join(data_dir, 'vqa/v2_mscoco_train2014_annotations.json')

        v_q_json_file = join(data_dir, 'vqa/v2_OpenEnded_mscoco_val2014_questions.json')
        v_a_json_file = join(data_dir, 'vqa/v2_mscoco_val2014_annotations.json')
    

    print("Loading Training questions")
    with open(t_q_json_file) as f:
        t_questions = json.loads(f.read())

    print("Loading Training answers")
    with open(t_a_json_file) as f:
        t_answers = json.loads(f.read())

    print("Loading Val questions")
    with open(v_q_json_file) as f:
        v_questions = json.loads(f.read())

    print("Loading Val answers")
    with open(v_a_json_file) as f:
        v_answers = json.loads(f.read())

    print("train|val (original ans)", len(t_answers['annotations']), len(v_answers['annotations']))
    print("train|val (original que)", len(t_questions['questions']), len(v_questions['questions']))

    answers = t_answers['annotations'] + v_answers['annotations']
    questions = t_questions['questions'] + v_questions['questions']

    answer_type_for_file = answer_type
    if answer_type == 'yes/no':
        answer_type_for_file = 'yesno'
    qa_data_file = join(data_dir, 'qa_v{}_{:4d}answers_{}.pkl'.format(version, num_ans, answer_type_for_file))
    vocab_file = join(data_dir, 'vocab_v{}_{:4d}answers_{}.pkl'.format(version, num_ans, answer_type_for_file))

    if answer_type in ('other', 'yes/no', 'number'):
        answers = [x for x in answers if x['answer_type']==answer_type]
        question_id_set = set([x['question_id'] for x in answers])
        questions = [q for q in questions if q['question_id'] in question_id_set]
 
    # find the top num_ans answers and their frequencies
    answer_vocab = make_answer_vocab(answers, num_ans)

    print(list(answer_vocab.keys())[:min(10, len(answer_vocab))])

    # find soft version of answers
    soft_answers = make_soft_answers(answer_vocab, answers)

    # find the most frequent words in questions and their frequencies, as well as the max_question_length
    question_vocab, max_question_length = make_questions_vocab(questions, answers, answer_vocab) 

    # only need words
    word_regex = re.compile(r'\w+')

    # qa data
    data = []
    for i,question in enumerate(questions):
        ans = answers[i]['multiple_choice_answer']
        ans = contract_word([ans])[0]
        # only need questions that has the top num_ans answers
        if ans in answer_vocab:
            data.append({
                'image_id' : answers[i]['image_id'],
                'question' : np.zeros(max_question_length),
                'answer' : answer_vocab[ans],
                'answers': soft_answers[answers[i]['question_id']]
                })
            question_sent = re.findall(word_regex, question['question'])
            question_sent = contract_word(question_sent)
            q_len = len(question_sent)
            data[-1]['ques_length'] = q_len
            for i in range(0, len(question_sent)):
                q_w = question_sent[i]
                data[-1]['question'][i] = question_vocab[ q_w ] \
                if q_w in question_vocab else question_vocab['UNK']

    print("nubmer of questions after filtering", len(data))
    print('answer_vocab', len(answer_vocab))
    print('question_vocab', len(question_vocab))
    print('max_question_length', max_question_length)

    # save all data in the `qa_data.pkl` file
    all_data = {
        'data' : data,
        'answer_vocab' : answer_vocab,
        'question_vocab' : question_vocab,
        'max_question_length' : max_question_length
    }

    print("Saving qa_data...")
    with open(qa_data_file, 'wb') as f:
        pickle.dump(all_data, f)

    # save vocabulary data in the `vocab_file2.pkl` file
    with open(vocab_file, 'wb') as f:
        vocab_data = {
        'answer_vocab' : all_data['answer_vocab'],
        'question_vocab' : all_data['question_vocab'],
        'max_question_length' : all_data['max_question_length']
        }
        pickle.dump(vocab_data, f)

    return data

def make_answer_vocab(answers, num_ans):
    
    answer_frequency = {} 
    for annotation in answers:
        answer = annotation['multiple_choice_answer']
        answer = contract_word([answer])[0]
        if answer in answer_frequency:
            answer_frequency[answer] += 1
        else:
            answer_frequency[answer] = 1

    answer_frequency_tuples = [ (-frequency, answer) for answer, frequency in answer_frequency.items()]
    answer_frequency_tuples.sort()
    top_n = min(num_ans, len(answer_frequency_tuples))
    answer_frequency_tuples = answer_frequency_tuples[0:top_n-1]

    answer_vocab = {}
    for i, ans_freq in enumerate(answer_frequency_tuples):
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
            this_ans = contract_word([an['answer']])[0]
            if this_ans in answer_vocab:
                a_list.append(this_ans)

        count = collections.Counter(a_list)
        for w, v in count.items():
            ans_dict[q_id][answer_vocab[w]] = v / float(len(a_list))
    return ans_dict

def make_questions_vocab(questions, answers, answer_vocab):
    word_regex = re.compile(r'\w+')
    question_frequency = {}
    max_question_length = 0
    for i,question in enumerate(questions):
        # answer for the question
        ans = answers[i]['multiple_choice_answer']
        ans = contract_word([ans])[0]
        count = 0

        # answer is among top num_ans
        if ans in answer_vocab:
            # tokenization
            question_words = re.findall(word_regex, question['question'])

        # update frequency for each token
        for qw in question_words:
            qw = contract_word([qw])[0]
            if qw in question_frequency:
                question_frequency[qw] += 1
            else:
                question_frequency[qw] = 1
            count += 1
    
        if count > max_question_length:
            max_question_length = count

    # set up the minimum frequency
    qw_freq_threhold = 0
    qw_tuples = [ (-frequency, qw) for qw, frequency in question_frequency.items()]
    # qw_tuples.sort() # takes too long
    qw_vocab = {}
    for i, qw_freq in enumerate(qw_tuples):
        frequency = -qw_freq[0]
        qw = qw_freq[1]
        if frequency > qw_freq_threhold:
            # +1 for accounting the zero padding for batch training
            qw_vocab[qw] = i + 1
        else:
            break

    qw_vocab['UNK'] = len(qw_vocab) + 1

    return qw_vocab, max_question_length

def load_questions_answers(data_dir='data', image_first=False, version=2, num_ans=1000):
    qa_data_file = join(data_dir, 'qa_v{}_{:4d}answers.pkl'.format(version, num_ans))
    if image_first:
        qa_data_file += '_imageFirst'

    data = pickle.load(open(qa_data_file, 'rb'))
    return data

def load_question_answer_vocab(data_dir='data', image_first=False, version=2, num_ans=1000):
    vocab_file = join(data_dir, 'vocab_v{}_{:4d}answers.pkl'.format(version, num_ans))
    if image_first:
        vocab_file += '_imageFirst'

    vocab_data = pickle.load(open(vocab_file, 'rb'))
    return vocab_data

# def load_cocoqa(data_dir='data'):
#     coco_dir = 'vqa/cocoqa/'

#     ans_vocab_path = os.path.join(data_dir, coco_dir, 'ansdict.pkl')
#     imgid_list_path = os.path.join(data_dir, coco_dir, 'imgid_dict.pkl')
#     ques_vocab_path = os.path.join(data_dir, coco_dir, 'qdict.pkl')
#     vocab_path = os.path.join(data_dir, coco_dir, 'vocab-dict.npy')

#     ans_vocab = pickle.load(open(ans_vocab_path, 'rb'), encoding='latin1')
#     ques_vocab = pickle.load(open(ques_vocab_path, 'rb'), encoding='latin1')
#     data = dict()
#     data['train'] = process_cocoqa('train')
#     data['val'] = process_cocoqa('valid')
#     data['answer_vocab'] = ans_vocab
#     data['question_vocab'] = ques_vocab
#     data['max_question_length'] = 55
#     return data

# def process_cocoqa(split='train'):
#     coco_dir = './data/vqa/cocoqa/'
#     if split == 'train':
#         data_path = 'train.npy'
#     elif split == 'valid':
#         data_path = 'valid.npy'
#     else:
#         print('Data split should be "train" for "valid".')

#     data = np.load(os.path.join(coco_dir, data_path), encoding='latin1')
#     ques_data = data[0]
#     ans_data = data[1]
#     assert(ques_data.shape[0] == ans_data.shape[0]), "Number of questions and number of answers not matching."
    
#     data = list()
#     for i in range(len(ques_data)):
#         temp = dict()
#         q = np.squeeze(ques_data[i])
#         temp['image_id'] = q[0]
#         temp['question'] = q[1:]
#         temp['answer'] = ans_data[i][0]
#         data.append(temp)

#     return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='save qa data')
    parser.add_argument('--num_answer', type=int, default=1000, help='number of answers (default=1000)')
    parser.add_argument('--version', type=int, default=2, help='vqa dataset version (1|2, default=2)')
    parser.add_argument('--answer_type', type=str, default='all', help='all|other|yesno|number (default=all)')
    args = parser.parse_args()

    prepare_training_data(version=args.version, num_ans=args.num_answer, answer_type=args.answer_type)
