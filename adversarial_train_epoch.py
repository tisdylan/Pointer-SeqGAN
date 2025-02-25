import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from dis_dataloader import get_dis_iter
from dis_collate_fn import dis_collate_fn

from gen_opts import gen_opts
from GAN_opts import GAN_opts
from cuda_opts import USE_CUDA

from checkpoint import save_gen_checkpoint

from gen_trainer_PG import gen_trainer_PG
from dis_train_epoch import train_dis

import numpy as np

from helper import PAD, SOS, EOS, UNK

import itertools

#from summarizer import summarizer


G_model_name = 'SeqGAN-Generator'
D_model_name = 'SeqGAN-Discriminator'
#observe_doc = "没有高考，你拼得过官二代吗？"

def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def print_statistics(epoch, num_epochs, num_iters, gen_iter, global_step, loss):
    
    print('='*100)
    print('Generator Training log:')
    print('- Epoch: {}/{}'.format(epoch, num_epochs))
    print('- Iter: {}/{}'.format(num_iters, gen_iter.__len__()))
    print('- Global step: {}'.format(global_step))
    print('- Loss: {}'.format(loss))

    print()
    
    # print("post:")
    # print(observe_doc)
    # print("summary:")
    # out_text = summarizer.summary(observe_doc)
    # print(out_text)
    
    # print('='*100 + '\n')


def save_checkpoint_training(encoder, decoder, encoder_optim, decoder_optim, epoch, num_iters, loss, global_step):
    savetime = ('%s' % datetime.now()).split('.')[0]
    experiment_name = '{}_{}'.format(G_model_name, savetime)

    checkpoint_path = save_gen_checkpoint(gen_opts, experiment_name, encoder, decoder, encoder_optim, 
                                          decoder_optim, epoch, num_iters, loss, global_step)
    
    print('='*100)
    print('Save checkpoint to "{}".'.format(checkpoint_path))
    print('='*100 + '\n')


def train_adversarial(dataset, generator, discriminator, encoder_optim, decoder_optim, dis_optim, gen_iter, gen_dataset, num_epochs, print_every_step, save_every_step, num_rollout):

    save_total_words = 0
    print_total_words = 0
    save_total_loss = 0
    print_total_loss = 0

    global_step = 0

    # print("BEFORE TRAINING")
    # print('-'*50)
    # print("post:")
    # print(observe_doc)
    # print('-'*50)
    # out_text = summarizer.summary(observe_doc)
    # print("summary:")
    # print(out_text)
    # print('='*100 + '\n')


    for epoch in range(1, num_epochs+1):
        for batch_id, batch_data in tqdm(enumerate(gen_iter)):
            
            print("\nStarting a G step...")
            # G steps

            # Unpack batch data
            src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens = batch_data

            # max_seq_len = max(src_lens)
            # if max_seq_len > gen_opts.max_seq_len:
            #     print('[!] Ignore batch: sequence length={} > max sequence length={}'.format(max_seq_len, gen_opts.max_seq_len))
            #     continue
            if len(src_lens) < gen_opts.batch_size: continue
            
            # # max_enc_seq_len
            max_enc_seq_len = 0
            for i in range(len(src_sents)):
                article_words = src_sents[i].split()
                # article_words = article_words[0]
                if len(article_words) > gen_opts.max_seq_len:
                    article_words = article_words[:gen_opts.max_seq_len]
                if len(article_words) > max_enc_seq_len:
                    max_enc_seq_len = len(article_words)
                
            # # enc_padding_mask
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

            del sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8, sub9, sub10, sub11, sub12, sub13, sub14, sub15, sub16, sub17, sub18, sub19

            ## Max Decoder Steps
            tgt_lens = torch.LongTensor(tgt_lens)
            tgt_lens_float = tgt_lens.float()
            if USE_CUDA:
                tgt_lens = tgt_lens.cuda()
                tgt_lens_float = tgt_lens_float.cuda()
            # max_dec_steps = int((int(tgt_lens_float.mean() * batch_size) - int(tgt_lens_float.min()) - int(tgt_lens_float.max()))/(batch_size-2) -15 )
            max_dec_steps = gen_opts.max_decoding_steps
            del tgt_lens_float

            # Get Decoder Input and Target Sequence
            def get_dec_inp_tgt_seq(sequence, max_len, start_id, stop_id):
                inp = [start_id] + sequence[:]
                target = sequence[:]
                if len(inp) > max_len: # truncate
                    inp = inp[:max_len]
                    target = target[:max_len] # no end_token
                else: # no truncation
                    target.append(stop_id) # end token
                assert len(inp) == len(target)
                return inp, target
            
            ## Abstract2IDS for abstract
            abs_ids_extend_vocab = [] # OOV are represented as a new id
            abs_ids = [] # OOV are represented as UNK
            for i in range(len(tgt_sents)):
                abstract_words = tgt_sents[i].split()
                abstract_words = [t for t in abstract_words if t != "" or t != "<s>" or t != "</s>"]  # remove empty and Start/End tags
                if len(abstract_words) > max_dec_steps:
                    abstract_words = abstract_words[:max_dec_steps]
                extend_ids = []
                ids = []
                for word in abstract_words:
                    word_id = dataset.vocab.token2id.get(word)
                    if word_id == UNK or word_id == None:  # If word is an OOV
                        if word in article_oovs_all[i]: # If word is an in-article OOV
                            vocab_idx = vocab_size + article_oovs_all[i].index(word) # Map to its temporary article OOV number
                            extend_ids.append(vocab_idx)
                            ids.append(UNK)
                        else: # If w is an out-of-article OOV
                            extend_ids.append(UNK) # Map to the UNK token id
                            ids.append(UNK)
                    else:
                        extend_ids.append(word_id)
                        ids.append(word_id)
                abs_ids_extend_vocab.append(extend_ids)
                abs_ids.append(extend_ids)
            
            # # Decoder Target Seqs
            # get_dec_inp_targ_seqs
            target_seq_all = []
            
            for i in range(len(tgt_sents)):
                _, target_seq = get_dec_inp_tgt_seq(abs_ids_extend_vocab[i], max_dec_steps, SOS, EOS)
                # inp = [SOS] + abs_ids_extend_vocab[i][:]
                # target_seq = abs_ids_extend_vocab[i][:]
                # if len(inp) > max_dec_steps: # truncate
                #     inp = inp[:max_dec_steps]
                #     target_seq = target_seq[:max_dec_steps] # no end_token
                # else: # no truncation
                #     target_seq.append(EOS) # end token
                # assert len(inp) == len(target_seq)
                target_seq_all.append(target_seq)
                # return inp, target_seq
            
            # Get dec_lens
            dec_lens = np.zeros((batch_size), dtype=np.int32)
            for i in range(len(tgt_sents)):
                dec_input, _ = get_dec_inp_tgt_seq(abs_ids[i], max_dec_steps, SOS, EOS)
                dec_lens[i] = len(dec_input)
            dec_lens = torch.from_numpy(dec_lens).float()
            dec_lens = get_cuda(dec_lens)
            # max_dec_len = np.max(dec_lens)

            # Get Target_batch
            target_batch = np.zeros((batch_size, max_dec_steps), dtype=np.int32)
            for i in range(len(tgt_sents)):
                target_seq_i = target_seq_all[i][:]
                target_batch[i, 0:len(target_seq_i)] = target_seq_i[:]
            target_batch = torch.from_numpy(target_batch).long()
            target_batch = get_cuda(target_batch) # CUDA
            del target_seq_i

            # # enc_batch_extend_vocab
            enc_batch_extend_vocab = None

            batch_enc_batch_extend_vocab = np.zeros((batch_size, max_enc_seq_len), dtype=np.int32)
            for i in range(len(src_sents)):
                enc_input_extend_vocab_i = enc_input_extend_vocab_all[i]
                batch_enc_batch_extend_vocab[i, 0:len(enc_input_extend_vocab_i)] = enc_input_extend_vocab_i[:]
            
            if batch_enc_batch_extend_vocab is not None:
                enc_batch_extend_vocab = torch.from_numpy(batch_enc_batch_extend_vocab).long()
                enc_batch_extend_vocab = get_cuda(enc_batch_extend_vocab)

            # # extra_zero
            extra_zeros = None
            max_art_oovs = 0

            for article_oovs in article_oovs_all:
                if len(article_oovs) > max_art_oovs:
                    max_art_oovs = len(article_oovs)
            # max_art_oovs=max([article_oovs in article_oovs_all])

            if max_art_oovs > 0:
                extra_zeros = torch.zeros(batch_size, max_art_oovs)
                extra_zeros = get_cuda(extra_zeros)

            # #context
            context = torch.zeros(batch_size, gen_opts.hidden_size*2)
            # context = torch.zeros(max_dec_steps, gen_opts.hidden_size)

            # 5. sum_temporal_srcs
            sum_temporal_srcs = None

            # 6. prev_s
            prev_s = None

            enc_lens = src_data_len # 替换 enc_lens

            src_seqs = src_data

            loss, num_words = gen_trainer_PG(src_seqs, src_lens, generator, encoder_optim, decoder_optim, enc_padding_mask, context, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s, extend_vocab_size, num_rollout=num_rollout, USE_CUDA=True)

            # Statistics.
            global_step += 1
            save_total_loss += loss
            print_total_loss += loss
            save_total_words += num_words
            print_total_words += num_words


            # Print statistics.
            if (batch_id + 1) % print_every_step == 0:
                print_loss = print_total_loss / print_total_words
                
                print_statistics(epoch, num_epochs, batch_id+1, gen_iter, global_step, print_loss)     
                
                print_total_loss = 0
                print_total_words = 0
                del print_loss

                # Save checkpoint.
            if (batch_id + 1) % save_every_step == 0:
                save_loss = save_total_loss / save_total_words
                    
                save_checkpoint_training(generator.encoder, generator.decoder, encoder_optim, decoder_optim, epoch, batch_id + 1, save_loss, global_step)
                    
                save_total_loss = 0
                save_total_words = 0
                
                del save_loss

            del src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, loss, num_words, enc_padding_mask, context, extra_zeros, enc_batch_extend_vocab, article_oovs_all, batch_enc_batch_extend_vocab, enc_input_extend_vocab_all, article_words
            
            torch.cuda.empty_cache()
                

            # D steps
            print("\nStarting a D step...")
            dis_iter = get_dis_iter(dataset = dataset, num_data=GAN_opts.batch_size * GAN_opts.d_step_repeat_times, num_workers=0)
            print("\nTraining discriminator...")
            train_dis(discriminator=discriminator,
                      dis_optim=dis_optim,
                      num_epochs=GAN_opts.dis_num_epoch,
                      dis_iter=dis_iter,
                      save_every_step=GAN_opts.D_save_every_step,
                      print_every_step=GAN_opts.D_print_every_step,
                      model_name=D_model_name)
    
    print("Finished adversarial training!")