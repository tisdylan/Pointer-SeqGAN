import os
import pickle
from tqdm import tqdm
from helper import REPO_DIR

from gen_dataset import gen_dataset
from gen_dataloader import get_gen_iter
from generator import generator

from gen_opts import LOAD_GEN_CHECKPOINT, gen_opts
import numpy as np
import torch

from cuda_opts import USE_CUDA
from helper import PAD, SOS, EOS, UNK

import itertools

def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor
    
def write_dis_dataset(dataset, gen_iter, generator, filepath):

    if not os.path.exists(REPO_DIR):
        os.makedirs(REPO_DIR)

    # iterator = enumerate(gen_iter)

    with open(filepath, "w") as f:
        for batch_id, batch_data in tqdm(enumerate(gen_iter)): 
            # batch_id, batch_data = next(iterator)
            # - sents:  [ [token] * seq_len ] * batch_size
            # - seqs:   (seq_len, batch_size)
            # - lens:   [1] * batch_size
            # Unpack batch data
            src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens = batch_data

            if len(src_lens) < gen_opts.batch_size: continue
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
            out_seqs, out_lens, decoder_outputs = generator(src_seqs, src_lens, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s, extend_vocab_size, enable_dropout=True, state=None)

            # (seq_len, batch_size) -> (batch_size, seq_len)
            src_seqs = src_seqs.t()         
            tgt_seqs = tgt_seqs.t()

            
            # - seq: (max_seq_len)
            real_pairs = list(zip(src_seqs.tolist(), tgt_seqs.tolist()))  # [(doc_seq, summary_seq)] * batch_size
            fake_pairs = list(zip(src_seqs.tolist(), out_seqs.tolist()))       # [(doc_seq, out_seq)] * batch_size


            out_str = []
            for doc, summary in real_pairs:
                doc_str = []
                for token_id in doc:
                    doc_str.append(str(token_id))
                summary_str = []
                for token_id in summary:
                    summary_str.append(str(token_id))

                out_str.append("\t".join([" ".join(doc_str), " ".join(summary_str)]) + "\t1\n")

            for doc, summary in fake_pairs:
                doc_str = []
                for token_id in doc:
                    doc_str.append(str(token_id))
                summary_str = []
                for token_id in summary:
                    summary_str.append(str(token_id))

                out_str.append("\t".join([" ".join(doc_str), " ".join(summary_str)]) + "\t0\n")

            f.writelines(out_str)

    print('='*100)
    print('Save file to "{}".'.format(filepath))
    print('='*100 + '\n')


def save_dis_dataset_pkl(filepath, tsv):

    def load_tsv(tsv):
        training_pairs = []

        for line in tqdm(open(tsv).read().strip().split('\n')):
            src_seqs, tgt_seqs, label = line.split('\t')

            src_seqs = [ int(seq) for seq in src_seqs.split() ]
            tgt_seqs = [ int(seq) for seq in tgt_seqs.split() ]
            label = int(label)

            training_pairs.append(((src_seqs, tgt_seqs), label))

        return training_pairs

    training_pairs = load_tsv(tsv)

    with open(filepath, "wb") as f:
        pickle.dump(training_pairs, f)

    return training_pairs


def load_dis_dataset_pkl(filepath):
    with open(filepath, "rb") as f:
        training_pairs = pickle.load(f)

    return training_pairs


def get_training_pairs(dataset):
    # For the sake of reusability, write into a file rather than generate negative example everytime

    TSV_FILE = os.path.join(REPO_DIR, "dis_tsv")
    PKL_FILE = os.path.join(REPO_DIR, "dis_pkl")

    # if os.path.exists(PKL_FILE):
    #     training_pairs = load_dis_dataset_pkl(PKL_FILE)
    # elif os.path.exists(TSV_FILE):
    #     training_pairs = save_dis_dataset_pkl(PKL_FILE, TSV_FILE)
    # else:
    #     gen_iter = get_gen_iter(gen_dataset=gen_dataset, batch_size=64, num_workers=4)
    #     write_dis_dataset(dataset, gen_iter, generator, TSV_FILE)
    #     training_pairs = save_dis_dataset_pkl(PKL_FILE, TSV_FILE)
    
    gen_iter = get_gen_iter(gen_dataset=gen_dataset, batch_size=20, num_workers=4)
    write_dis_dataset(dataset, gen_iter, generator, TSV_FILE)
    training_pairs = save_dis_dataset_pkl(PKL_FILE, TSV_FILE)

    return training_pairs