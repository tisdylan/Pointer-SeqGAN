#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import Counter
import os, pickle, random, glob, struct

from tensorflow.core.example import example_pb2

from helper import REPO_DIR
from AttrDict import AttrDict
from helper import PAD, SOS, EOS, UNK

class GenDataset(Dataset):
    def __init__(self, training_pairs, vocab=None, counter=None, filter_vocab=True, max_vocab_size=10000):
        """ Note: If src_vocab, tgt_vocab is not given, it will build both vocabs.
            Args: 
            - src_path, tgt_path: text file with tokenized sentences.
            - src_vocab, tgt_vocab: data structure is same as self.build_vocab().
        """
        print('='*100)
        print('- Loading and tokenizing training sentences...')

        self.src_sents, self.tgt_sents = self.load_sents(training_pairs)
        print ("Sentences Loaded.")
    
        if vocab is None:
            if counter is None:
                print('- Building source counter...')
                self.counter = self.build_counter(training_pairs)
            else:
                self.counter = counter
            print('- Building source vocabulary...')
            self.vocab = self.build_vocab(self.counter, filter_vocab, max_vocab_size)
            print('- Saving vocab...')
            save_vocab(self.vocab)
        else:
            self.vocab = vocab

        print('='*100)
        print('Dataset Info:')
        print('- Number of training pairs: {}'.format(self.__len__()))
        print('- Vocabulary size: {}'.format(len(self.vocab.token2id)))
        print('='*100 + '\n')
    
    def __len__(self):
        return len(self.src_sents) # 返回 source_sentences length
    
    def __getitem__(self, index):
        src_sent = self.src_sents[index]
        tgt_sent = self.tgt_sents[index]
        src_seq = self.tokens2ids(src_sent, self.vocab.token2id, append_SOS=False, append_EOS=True)
        tgt_seq = self.tokens2ids(tgt_sent, self.vocab.token2id, append_SOS=False, append_EOS=True)

        return src_sent, tgt_sent, src_seq, tgt_seq
    
    def load_sents(self, sent_pairs):
        print ("Loding sentences...")
        src_sents, tgt_sents = zip(*sent_pairs)
        return src_sents, tgt_sents

    def build_counter(self, sents):
        counter = Counter()
        words = []
        for i in tqdm(range(len(sents))):
        #for art_sent, abs_sent in tqdm(sents): #在这里拆词？
            art_sent = sents[i][0] #temp
            abs_sent = sents[i][1] #temp
            art_words = art_sent.split() #你怎么坏了?
            abs_words = abs_sent.split()
            ##abs_words = [t for t in abs_words if t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
            words = words + art_words + abs_words

            # if i == 1:
            #     print(words)
        
        words = [t.strip() for t in words]
        words = [t for t in words if t != "" or t != "<s>" or t != "</s>"]  # remove empty
        # words = [t for t in words if t != "<s>"]  ## remove <s>
        # words = [t for t in words if t != "</s>"]  ## remove </s>
        for word in words:
            counter[word] += 1
        # counter.update(words)
        ## counter.update(r_sent)
        # print('='*100)
        # print(len(words))
        # # print(counter.__len__)
        # print(len(counter))
        # print('='*100)
        del words, art_words, abs_words
        return counter
    
    def build_vocab(self, counter, filter_vocab, max_vocab_size):
        vocab = AttrDict()
        vocab.token2id = {'<PAD>': PAD, '<SOS>': SOS, '<EOS>': EOS, '<UNK>': UNK}
        
        if filter_vocab:
            # for _id, (token, count) in tqdm(enumerate(counter.most_common(max_vocab_size-4))):
            #     vocab.token2id.update({token: _id+4})
            vocab.token2id.update({token: _id+4 for _id, (token, count) in tqdm(enumerate(counter.most_common(max_vocab_size-4)))})
        else:
            vocab.token2id.update({token: _id+4 for _id, (token, count) in tqdm(enumerate(counter.most_common()))})
        
        vocab.id2token = {v:k for k,v in tqdm(vocab.token2id.items())}

        return vocab
    
    def tokens2ids(self, tokens, token2id, append_SOS=True, append_EOS=True):
        seq = []
        oovs=[]
        vocab_size = len(self.vocab.token2id)
        if append_SOS: 
            seq.append(SOS)
        words = tokens[0].split()
        words = [t.strip() for t in words]
        words = [t for t in words if t != "" or t != "</s>" or t != "<s>"]  # remove empty/<s>/</s>
        for w in words:
            # i = 0
            i = token2id.get(w)
            if i == UNK or i == None:
                i == UNK
            else:
                seq.append(i)
        if append_EOS: 
            seq.append(EOS)
        return seq

    # def get_article_oovs(article_words, vocab):
    #     ids = []
    #     oovs = []
    #     unk_id = vocab.token2id(UNK)
    #     for w in article_words:
    #         i = vocab.word2id(w)
    #         if i == unk_id: # If w is OOV
    #             if w not in oovs: # Add to list of OOVs
    #                 oovs.append(w)
    #             oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
    #             ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
    #         else:
    #             ids.append(i)
    #     return ids, oovs
    
def text_generator(example_generator):
    print ("Text Generator: getting article or abstract from example...")
    while True:
        e = next(example_generator) # e is a tf.Example
        try:
            article_text = e.features.feature['article'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
            abstract_text = e.features.feature['abstract'].bytes_list.value[0] # the abstract text was saved under the key 'abstract' in the data files
            article_text = article_text.decode()
            abstract_text = abstract_text.decode()
        except ValueError:
            print('Error: Failed to get article or abstract from example')
            continue
        if len(article_text)==0: # See https://github.com/abisee/pointer-generator/issues/1
            print('Warning: Found an example with empty article text. Skipping it.')
            continue
        else:
            yield (article_text, abstract_text)

def example_generator(data_path, single_pass):
# Generates tf.Examples from data files.
# Binary data format: <length><blob>. <length> represents the byte size
# of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
# the tokenized article text and summary.
#       Args:
#         data_path:
#           Path to tf.Example data files. Can include wildcards, e.g. if you have several training data chunk files train_001.bin, train_002.bin, etc, then pass data_path=train_* to access them all.
#         single_pass:
#           Boolean. If True, go through the dataset exactly once, generating examples in the order they appear, then return. Otherwise, generate random examples indefinitely.
#       Yields:
#         Deserialized tf.Example.

    print ("Example Generator: Loading filelist...")
    # while True:
    filelist = glob.glob(data_path) # get the list of datafiles
        #print (filelist)
    assert filelist, ('Example Generator: Error: Empty filelist at %s' % data_path) # check filelist isn't empty # 有 filelist, 否则返回错误
    if single_pass:
        filelist = sorted(filelist)
    else:
        print ("Example Generator: Shuffling filelist...")
        random.shuffle(filelist)
        print ("Example Generator: Filelist Shuffled.")
    
    print ("Example Generator: Reading filelist...")
    for f in filelist:
        reader = open(f, 'rb') # 读取数据
        while True:
            len_bytes = reader.read(8)
            if not len_bytes: break # finished reading this file
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            #yield example_pb2.Example.FromString(example_str)
    # if single_pass:
    #     print ("example_generator completed reading all datafiles. No more data.")
    #     break

## -----------------------------------------------------------------------------------

from GAN_opts import GAN_opts
from gen_opts import gen_opts

import pandas as pd
from tqdm import trange

import csv
import numpy as np

def save_vocab(vocab):
    np.save('vocab.npy', vocab)

def load_vocab(vocab_url):
    from pathlib import Path
    vocab_url = Path(vocab_url)
    if vocab_url.is_file():
        vocab = np.load(vocab_url, allow_pickle=True).item()
    else:
        print("Vocab file not found. GenDataset will build one instead.")
        vocab = None
    return vocab

count = 0

training_pairs = []
src_sents = []
tgt_sents = []

# input_gen_df = pd.read_csv(r'/usr/local/Convolutional SeqGAN/small-covid-19.csv')
# input_gen_df = pd.read_csv(r'/usr/local/Convolutional SeqGAN/small-250.csv')
input_gen_df = pd.read_csv(r'./covid-19-withoutEmpty.csv')
# input_gen_df = pd.read_csv(r'./covid-19-1.csv')

input_gen_df.sample(frac=1)
input_gen_df.info()

for i in trange(len(input_gen_df)):
    tgt_sent = str(input_gen_df.iloc[i,0]).strip()
    src_sent = str(input_gen_df.iloc[i,1]).strip()
    
    tgt_sents.append(tgt_sent) #TITLE
    src_sents.append(src_sent) #ABSTRACT

    del tgt_sent, src_sent
    count += 1

print("Count: ", count)

print ("Getting training pairs...")
training_pairs=(list(zip(src_sents, tgt_sents)))
print("Training Pairs Length: ", len(training_pairs))

vocab = load_vocab('./vocab-50000-fixed.npy')

def get_gen_dataset(num_data=None):
    if num_data:
        assert(num_data <= len(training_pairs))
        num_data = int(num_data)
        gen_dataset = GenDataset(vocab = vocab,
                                 training_pairs=random.sample(training_pairs, num_data),
                                 filter_vocab=gen_opts.filter_vocab,
                                 max_vocab_size=gen_opts.max_vocab_size)        
    else:
        gen_dataset = GenDataset(vocab = vocab,
                                 training_pairs=training_pairs,
                                 filter_vocab=gen_opts.filter_vocab,
                                 max_vocab_size=gen_opts.max_vocab_size)
    
    return gen_dataset

gen_dataset = get_gen_dataset()
