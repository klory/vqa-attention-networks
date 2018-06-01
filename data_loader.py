import json
import argparse
from os.path import isfile, join
import re
import numpy as np
import pprint
import pickle
import collections
import pdb
import torch
import torch.utils.data as data
from os.path import join

num_ans = 5216

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
articles     = ['a',
                'an',
                'the'
              ]

def contract_word(word):
  word = word.lower()
  if word in contractions:
    return contractions[word]
  if word in manualMap:
    return manualMap[word]
  if word in articles:
    return 'a'
  return word

def prepare_training_data(data_dir = 'data', image_first=False):
  t_q_json_file = join(data_dir, 'vqa/v2_OpenEnded_mscoco_train2014_questions.json')
  t_a_json_file = join(data_dir, 'vqa/v2_mscoco_train2014_annotations.json')

  v_q_json_file = join(data_dir, 'vqa/v2_OpenEnded_mscoco_val2014_questions.json')
  v_a_json_file = join(data_dir, 'vqa/v2_mscoco_val2014_annotations.json')
  
  qa_data_file = join(data_dir, 'qa_data_file2.pkl')
  vocab_file = join(data_dir, 'vocab_file2.pkl')

  if image_first:
    qa_data_file = join(data_dir, 'qa_data_file2_image_first.pkl')
    vocab_file = join(data_dir, 'vocab_file2_image_first.pkl')

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

  print("Ans", len(t_answers['annotations']), len(v_answers['annotations']))
  print("Qu", len(t_questions['questions']), len(v_questions['questions']))

  answers = t_answers['annotations'] + v_answers['annotations']
  questions = t_questions['questions'] + v_questions['questions']

  # find the top num_ans answers and their frequencies
  answer_vocab = make_answer_vocab(answers)

  # find soft version of answers
  soft_answers = make_soft_answers(answer_vocab, answers)

  # find the most frequent words in questions and their frequencies, as well as the max_question_length
  question_vocab, max_question_length = make_questions_vocab(questions, answers, answer_vocab) 
  print("Question Vocabulary Size", len(question_vocab))
  print("Max Question Length", max_question_length)

  # only need words
  word_regex = re.compile(r'\w+')
  training_data = []
  for i,question in enumerate( t_questions['questions']):
    ans = t_answers['annotations'][i]['multiple_choice_answer']
    ans = contract_word(ans)
    # only need questions that has the top num_ans answers
    if ans in answer_vocab:
      training_data.append({
        'image_id' : t_answers['annotations'][i]['image_id'],
        'question' : np.zeros(max_question_length),
        'answer' : answer_vocab[ans],
        'answers': soft_answers[t_answers['annotations'][i]['question_id']]
        })
      question_words = re.findall(word_regex, question['question'])
      base = max_question_length - len(question_words)
      if not image_first:
        for i in range(0, len(question_words)):
          q_w = contract_word(question_words[i])
          training_data[-1]['question'][base + i] = question_vocab[ q_w ] \
          if q_w in question_vocab else question_vocab['UNK']
      else:
        for i in range(0, len(question_words)):
          q_w = contract_word(question_words[i])
          training_data[-1]['question'][i] = question_vocab[ q_w ] \
          if q_w in question_vocab else question_vocab['UNK']

  print("Training Data", len(training_data))
  
  val_data = []
  for i,question in enumerate( v_questions['questions']):
    ans = v_answers['annotations'][i]['multiple_choice_answer']
    ans = contract_word(ans)
    if ans in answer_vocab:
      val_data.append({
        'image_id' : v_answers['annotations'][i]['image_id'],
        'question' : np.zeros(max_question_length),
        'answer' : answer_vocab[ans],
        'answers': soft_answers[v_answers['annotations'][i]['question_id']]
        })
      question_words = re.findall(word_regex, question['question'])
      base = max_question_length - len(question_words)
      if not image_first:
        for i in range(0, len(question_words)):
          q_w = contract_word(question_words[i])
          training_data[-1]['question'][base + i] = question_vocab[ q_w ] \
          if q_w in question_vocab else question_vocab['UNK']
      else:
        for i in range(0, len(question_words)):
          q_w = contract_word(question_words[i])
          training_data[-1]['question'][i] = question_vocab[ q_w ] \
          if q_w in question_vocab else question_vocab['UNK']

  print("Validation Data", len(val_data))

  # save all data in the `qa_data.pkl` file
  data = {
    'train' : training_data,
    'val' : val_data,
    'answer_vocab' : answer_vocab,
    'question_vocab' : question_vocab,
    'max_question_length' : max_question_length
  }

  print("Saving qa_data")
  with open(qa_data_file, 'wb') as f:
    pickle.dump(data, f)

  # save all data in the `vocab_file2.pkl` file
  with open(vocab_file, 'wb') as f:
    vocab_data = {
      'answer_vocab' : data['answer_vocab'],
      'question_vocab' : data['question_vocab'],
      'max_question_length' : data['max_question_length']
    }
    pickle.dump(vocab_data, f)

  return data

def make_answer_vocab(answers):
  top_n = num_ans
  answer_frequency = {} 
  for annotation in answers:
    answer = annotation['multiple_choice_answer']
    answer = contract_word(answer)
    if answer in answer_frequency:
      answer_frequency[answer] += 1
    else:
      answer_frequency[answer] = 1

  answer_frequency_tuples = [ (-frequency, answer) for answer, frequency in answer_frequency.items()]
  answer_frequency_tuples.sort()
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
      this_ans = contract_word(an['answer'])
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
    ans = contract_word(ans)
    count = 0

    # answer is among top num_ans
    if ans in answer_vocab:
      # tokenization
      question_words = re.findall(word_regex, question['question'])

      # update frequency for each token
      for qw in question_words:
        qw = contract_word(qw)
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

def load_questions_answers(data_dir='data'):
  qa_data_file = join(data_dir, 'qa_data_file2.pkl')
  data = pickle.load(open(qa_data_file, 'rb'))
  return data

def get_question_answer_vocab(data_dir='data'):
  vocab_file = join(data_dir, 'vocab_file2.pkl')
  vocab_data = pickle.load(open(vocab_file, 'rb'))
  return vocab_data

if __name__ == '__main__':
  prepare_training_data(image_first=True)