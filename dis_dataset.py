import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np
import random
from gen_opts import LOAD_GEN_CHECKPOINT, gen_opts
from cuda_opts import USE_CUDA
from helper import PAD, SOS, EOS, UNK
import itertools

def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

class DisDataset(Dataset):
    def __init__(self, dataset = None, training_pairs=None, gen_iter=None, generator=None):
        
        if training_pairs is not None:
            self.dialogue_pairs, self.labels = self.load_data_by_training_pairs(training_pairs)
        else:
            self.dialogue_pairs, self.labels = self.load_data_by_generator(dataset, gen_iter, generator)

        print('='*100)
        print('Dataset Info:')
        print('- Number of training pairs: {}'.format(self.__len__()))
        print('='*100 + '\n')


    def __len__(self):
        return len(self.labels)
    

    def __getitem__(self, index):
        doc_seq, summary_seq = self.dialogue_pairs[index]
        label = self.labels[index]

        # - doc_seq:      (seq_len)
        # - summary_seq:   (seq_len)
        # - label:          int, 1 or 0
        return doc_seq, summary_seq, label


    def load_data_by_training_pairs(self, training_pairs):
        dialogue_pairs, labels = [], []
        for (src_seqs, tgt_seqs), label in tqdm(training_pairs):
            
            src_seqs = torch.LongTensor(src_seqs)
            tgt_seqs = torch.LongTensor(tgt_seqs)
            
            dialogue_pairs.append((src_seqs, tgt_seqs))
            labels.append(label)

        return dialogue_pairs, labels


    def load_data_by_generator(self, dataset, gen_iter, generator):
        dialogue_pairs, labels = [], []
        for batch_id, batch_data in tqdm(enumerate(gen_iter)):
            # - sents:  [ [token] * seq_len ] * batch_size
            # - seqs:   (seq_len, batch_size)
            # - lens:   [1] * batch_size
            src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens = batch_data

            if len(src_lens) < gen_opts.batch_size: continue

            batch_size = src_seqs.size(1)
            # 1. enc_padding_mask
            # 1.1 max_enc_seq_len
            max_enc_seq_len = 0
            for i in range(len(src_sents)):
                article_words = src_sents[i].split()
                # article_words = article_words[0]
                if len(article_words) > gen_opts.max_seq_len:
                    article_words = article_words[:gen_opts.max_seq_len]
                if len(article_words) > max_enc_seq_len:
                    max_enc_seq_len = len(article_words)
                
            # 1.2 enc_padding_mask
            batch_size = gen_opts.batch_size
            enc_lens = np.zeros((batch_size), dtype=np.int32)
            enc_padding_mask = np.zeros((batch_size, max_enc_seq_len), dtype=np.float32)
            enc_padding_mask = torch.from_numpy(enc_padding_mask).float()

            for i in range(len(src_sents)):
                article_words = src_sents[i].split()
                # article_words = article_words[0].split()
                if len(article_words) > gen_opts.max_seq_len:
                    article_words = article_words[:gen_opts.max_seq_len]
                enc_lens[i] = len(article_words)
                for j in range(len(article_words)):
                    enc_padding_mask[i][j] = 1
            
            # 2. ct_e (aka context)
            ct_e = torch.zeros(batch_size, 2*gen_opts.hidden_size)

            # 3. extra_zeros
            ## 3.1 Articles2IDs
            # Articles2IDs for article
            enc_input_extend_vocab_all = []
            article_oovs_all=[]
            vocab_size = len(dataset.vocab.token2id)

            src_data = [] # 建立一个新的列表, 用来存储真正的 sents 的 id, 用来给 encoder/decoder 输入
            src_data_len = [] # 记录长度

            extend_vocab_size = vocab_size # Extended vocab 最大值
            for i in range(len(src_sents)):
                article_words = src_sents[i].split()
                # article_words = article_words[0]
                if len(article_words) > gen_opts.max_seq_len:
                    article_words = article_words[:gen_opts.max_seq_len]
                ids=[]
                oovs=[]
                limited_ids=[]
                for word in article_words:
                    word_id = dataset.vocab.token2id.get(word)
                    if word_id == UNK or word_id == None:  # If word is an OOV
                        if word not in oovs:
                            oovs.append(word) # Add word in oovs
                        oov_num = oovs.index(word) # word 在 oovs 中的序号
                        ids.append(vocab_size+oov_num) # 每一轮都应该从 vocab_size 开始数
                        limited_ids.append(UNK) # 一个严格的无超出词表的 id 列表, 如果超出词表, 用 UNK, 避免 LongTensor 中出现 none
                    else:
                        ids.append(word_id)
                        limited_ids.append(word_id) # 一个严格的无超出词表的 id 列表
                extend_vocab_size = max(vocab_size+len(oovs), extend_vocab_size) #记录最大 vocab size
                enc_input_extend_vocab_all.append(ids)
                article_oovs_all.append(oovs)
                
                src_data.append(limited_ids) # 每一行都是列表, 每一个列表里面是独立的字符对应的 id
                src_data_len.append(len(limited_ids)) #每一行是一个数字, 记录 id 长度

                del oovs, limited_ids
            
            # 获得子列表, 否则无法 zip
            sub0 = src_data[0]
            sub1 = src_data[1]
            sub2 = src_data[2]
            sub3 = src_data[3]
            sub4 = src_data[4]
            sub5 = src_data[5]
            sub6 = src_data[6]
            sub7 = src_data[7]
            sub8 = src_data[8]
            sub9 = src_data[9]
            sub10 = src_data[10]
            sub11 = src_data[11]
            sub12 = src_data[12]
            sub13 = src_data[13]
            sub14 = src_data[14]
            sub15 = src_data[15]
            sub16 = src_data[16]
            sub17 = src_data[17]
            sub18 = src_data[18]
            sub19 = src_data[19]

            src_data = list(itertools.zip_longest(sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8, sub9, sub10, sub11, sub12, sub13, sub14, sub15, sub16, sub17, sub18, sub19, fillvalue=PAD))
            src_data = torch.LongTensor(src_data) # 转成 longTensor

            # 取消替换, 因为下面要用到 src_seqs
            src_seqs = src_data
            src_lens = src_data_len # 替换 esrc_lens

            del sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8, sub9, sub10, sub11, sub12, sub13, sub14, sub15, sub16, sub17, sub18, sub19
            
            ## 3.2 extra_zeros
            extra_zeros = None
            max_art_oovs = 0

            for article_oovs in article_oovs_all:
                if len(article_oovs) > max_art_oovs:
                    max_art_oovs = len(article_oovs)
            if max_art_oovs > 0:
                extra_zeros = torch.zeros(batch_size, max_art_oovs)
                extra_zeros = get_cuda(extra_zeros)

            # 4. enc_batch_extend_vocab
            enc_batch_extend_vocab = None

            batch_enc_batch_extend_vocab = np.zeros((batch_size, max_enc_seq_len), dtype=np.int32)
            for i in range(len(src_sents)):
                enc_input_extend_vocab_i = enc_input_extend_vocab_all[i]
                batch_enc_batch_extend_vocab[i, 0:len(enc_input_extend_vocab_i)] = enc_input_extend_vocab_i[:]
            
            if batch_enc_batch_extend_vocab is not None:
                enc_batch_extend_vocab = torch.from_numpy(batch_enc_batch_extend_vocab).long()
                enc_batch_extend_vocab = get_cuda(enc_batch_extend_vocab)
            
            # 5. sum_temporal_srcs
            sum_temporal_srcs = None

            # 6. prev_s
            prev_s = None

            # - out_seqs:           (batch_size, max_out_len)
            # - out_lens:           (batch_size)
            # - decoder_outputs:    (batch_size, max_out_len, vocab_size)
            # out_seqs, out_lens, decoder_outputs = generator(src_seqs, src_lens, enable_dropout=False)
            out_seqs, out_lens, decoder_outputs = generator(src_data, src_data_len, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s, extend_vocab_size, enable_dropout=False, state=None)

            ################ tgt_seqs ################
            #enc_input_extend_vocab_all = []
            #abstract_oovs_all=[]
            vocab_size = len(dataset.vocab.token2id)

            tgt_data = [] # 建立一个新的列表, 用来存储真正的 tgt_sents 的 id, 用来给 encoder/decoder 输入
            tgt_data_len = [] # 记录长度

            extend_vocab_size = vocab_size # Extended vocab 最大值
            for i in range(len(tgt_sents)):
                abstract_words = tgt_sents[i].split()
                if len(abstract_words) > gen_opts.max_decoding_steps:
                    abstract_words = abstract_words[:gen_opts.max_decoding_steps]
                #ids=[]
                #oovs=[]
                limited_ids=[] # 一个严格的无超出词表的 id 列表, 如果超出词表, 用 UNK, 避免 LongTensor 中出现 none
                for word in abstract_words:
                    word_id = dataset.vocab.token2id.get(word)
                    if word_id == UNK or word_id == None:  # If word is an OOV
                        #if word not in oovs:
                        #    oovs.append(word) # Add word in oovs
                        #oov_num = oovs.index(word) # word 在 oovs 中的序号
                        #ids.append(vocab_size+oov_num) # 每一轮都应该从 vocab_size 开始数
                        limited_ids.append(UNK) # 一个严格的无超出词表的 id 列表, 如果超出词表, 用 UNK, 避免 LongTensor 中出现 none
                    else:
                        #ids.append(word_id)
                        limited_ids.append(word_id) # 一个严格的无超出词表的 id 列表
                #extend_vocab_size = max(vocab_size+len(oovs), extend_vocab_size) #记录最大 vocab size
                #enc_input_extend_vocab_all.append(ids)
                #abstract_oovs_all.append(oovs)
                
                tgt_data.append(limited_ids) # 每一行都是列表, 每一个列表里面是独立的字符对应的 id
                tgt_data_len.append(len(limited_ids)) #每一行是一个数字, 记录 id 长度

                #del oovs, limited_ids
                del limited_ids
            
            # 获得子列表, 否则无法 zip
            sub0 = tgt_data[0]
            sub1 = tgt_data[1]
            sub2 = tgt_data[2]
            sub3 = tgt_data[3]
            sub4 = tgt_data[4]
            sub5 = tgt_data[5]
            sub6 = tgt_data[6]
            sub7 = tgt_data[7]
            sub8 = tgt_data[8]
            sub9 = tgt_data[9]
            sub10 = tgt_data[10]
            sub11 = tgt_data[11]
            sub12 = tgt_data[12]
            sub13 = tgt_data[13]
            sub14 = tgt_data[14]
            sub15 = tgt_data[15]
            sub16 = tgt_data[16]
            sub17 = tgt_data[17]
            sub18 = tgt_data[18]
            sub19 = tgt_data[19]

            tgt_data = list(itertools.zip_longest(sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8, sub9, sub10, sub11, sub12, sub13, sub14, sub15, sub16, sub17, sub18, sub19, fillvalue=PAD))
            tgt_data = torch.LongTensor(tgt_data) # 转成 longTensor

            tgt_seqs = tgt_data
            #tgt_lens = tgt_data_len # 用不到

            del sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8, sub9, sub10, sub11, sub12, sub13, sub14, sub15, sub16, sub17, sub18, sub19
            #######################################

            # (seq_len, batch_size) -> (batch_size, seq_len)
            src_seqs = src_seqs.t()
            tgt_seqs = tgt_seqs.t()

            # - seq: (max_seq_len)
            real_pairs = list(zip(src_seqs, tgt_seqs))  # [(doc_seq, summary_seq)] * batch_size
            fake_pairs = list(zip(src_seqs, out_seqs))       # [(doc_seq, out_seq)] * batch_size

            dialogue_pairs.extend(real_pairs)
            labels.extend([1] * batch_size)
            
            dialogue_pairs.extend(fake_pairs)
            labels.extend([0] * batch_size)

        return dialogue_pairs, labels