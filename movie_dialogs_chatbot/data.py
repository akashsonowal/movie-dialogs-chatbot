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
















