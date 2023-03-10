from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals 

import torch 
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv 
import random 
import re 
import os 
import unicodedata
import codecs
from io import open
import itertools 
import math 
import json 

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "movie_corpus"
corpus = os.path.join("data", corpus_name)

def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

printLines(os.path.join(corpus, "utterances.jsonl"))

def loadLinesAndConversations(fileName):
    lines = {}
    conversations = {}

    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            lineJson = json.loads(line)
            #lineObj gets in lines dict
            lineObj = {}
            lineObj['lineID'] = lineJson['id']
            lineObj['characterID'] = lineJson['speaker']
            lineObj['text'] = lineJson['text']

            lines[lineObj['lineID']] = lineObj 

            if lineJson['conversation_id'] not in conversations:
                convObj = {}
                convObj['conversationID'] = lineJson['conversation_id']
                convObj['movieID'] = lineJson['meta']['movie_id']
                convObj['lines'] = [lineObj]
            else:
                convObj = conversations[lineJson['conversation_id']]
                convObj['lines'].insert(0, lineObj)
            conversations[convObj['conversationID']] = convObj 
    return lines, conversations 

def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations.values():
        for i in range(len(conversation) - 1): #last line has no answer
            inputLine = conversation['lines'][i]['text'].strip()
            targetLine = conversation['lines'][i + 1]['text'].strip()
            
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs 

datafile = os.path.join(corpus, 'formatted_movie_lines.txt')

delimiter = "\t"
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

lines = {}
conversations = {}
print('\n Processing corpus into lines and conversations...')
lines, conversations = loadLinesAndConversations(os.path.join(corpus, "utterances.jsonl"))

#Writing new formatted file
print("\n Writing new formatted file...")
with open(datafile, 'w', encoding='utf-8') as outfile:
    writer = csv.writer(outfile, delimiter=delimiter, lineterminator='\n')
    for pair in conversations:
        writer.writerow(pair)

print('\n Sample lines from file:')
printLines(datafile)

PAD_token = 0
SOS_token = 1
EOS_token = 2

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False 
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3
    
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
    
    def trim(self, min_count):
        if self.trimmed:
            return 
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        print(f'keep words {len(keep_words)} / {len(self.word2index)} = {(len(keep_words)/len(self.word2index)):.4f}')
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: ['PAD'], SOS_token: ['SOS'], EOS_token: ['EOS']}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)
    
MAX_LENGTH = 10
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != Mn
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]', r' ', s)
    s = re.sub(r'\s+', r' ', s).strip()
    return s 

def readVocs(datafile, corpus_name):
    print('Reading lines...')
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs 

def filterPair(p):
    return len(p[0].split()) < MAX_LENGTH and len(p[1].split()) < MAX_LENGTH

def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print('Start preparing training data...')
    voc, pairs = readVocs(datafile, corpus_name)
    print(f'Read {len(pairs)} sentence pairs')
    pairs = filterPair(pairs)
    print(f'Trimmed to {len(pairs)} sentence pairs')
    print(f'Counting words...')
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print(f'Counted words {voc.num_words}')
    return voc, pairs

save_dir = os.path.join('data', 'save')
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
print('\n pairs:')
for pair in pairs[:10]:
    print(pair)

MIN_COUNT = 3

def trimRareWords(voc, pairs, MIN_COUNT):
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False 
                break 
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False 
                break 
        
        if keep_input and keep_output: 
            keep_pairs.append(pair) 
        
    print(f'Trimmed from {len(pairs)} to {len(keep_pairs), {(len(pairs) / len(keep_pairs))}:.4f} of total')
    return keep_pairs 

pairs = trimRareWords(voc, pairs, MIN_COUNT)


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m 

def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.Tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len 

def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len 

small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print('input_variable', input_variable)
print('lengths', lengths)
print('target_variable', target_variable)
print('mask', mask)
print('max_target_len', max_target_len)


























