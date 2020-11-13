import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F
from train_util import get_cuda

from cuda_opts import device

#config
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4

def init_lstm_wt(lstm):
    for name, _ in lstm.named_parameters():
        if 'weight' in name:
            wt = getattr(lstm, name)
            wt.data.uniform_(-rand_unif_init_mag, rand_unif_init_mag)
        elif 'bias' in name:
            # set forget bias to 1
            bias = getattr(lstm, name)
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data.fill_(0.)
            bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=trunc_norm_init_std)

def init_wt_normal(wt):
    wt.data.normal_(std=trunc_norm_init_std)

class Encoder(nn.Module):
    def __init__(self, embedding, hidden_size=512, num_layers=1, dropout=0.3, batch_first =True, bidirectional=True, fixed_embeddings=False):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # self.hidden_size = hidden_size // self.num_directions
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        self.embedding.weight.data.copy_(embedding.weight)
        if fixed_embeddings:
            self.embedding.weight.requires_grad = False
        
        self.word_vec_size = self.embedding.embedding_dim

        # self.lstm = nn.LSTM(256, 512, num_layers=1, batch_first=True, bidirectional=True)
        # init_lstm_wt(self.lstm)

        self.rnn_type = 'LSTM'
        self.rnn = getattr(nn, self.rnn_type)(
                           input_size=self.word_vec_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           dropout=self.dropout, 
                           bidirectional=self.bidirectional)
        
        # self.init_parameters()

        # self.reduce_h = nn.Linear(1024, 512)
        self.reduce_h = nn.Linear(512, 256)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(512, 256)
        # self.reduce_c = nn.Linear(1024, 512)
        init_linear_wt(self.reduce_c)

    def forward(self, src_seqs, seq_lens): 
        src_lens = seq_lens
        emb = self.embedding(src_seqs)
        batch_size = len(seq_lens)
        enc_hid = self.initHidden(batch_size)
        enc_out, enc_hid = self.rnn(emb, enc_hid) # (max_seq_len, batch, hidden_size), (num_layers * num_directions, batch_size, hidden_size)

        # packed = pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        # enc_out, enc_hid = self.lstm(packed)
        # enc_out,_ = pad_packed_sequence(enc_out, batch_first=True)
        enc_out = enc_out.contiguous()                              #bs, n_seq, 2*n_hid
        enc_out = enc_out.view(batch_size, 120, 512)                #bs, n_seq, 2*n_hid # 手动 resize ?
        h, c = enc_hid                                              #shape of h: 2, bs, n_hid
        h = torch.cat(list(h), dim=1)                               #bs, 2*n_hid
        c = torch.cat(list(c), dim=1)
        h_reduced = F.relu(self.reduce_h(h))                        #bs,n_hid
        c_reduced = F.relu(self.reduce_c(c))
        return enc_out, (h_reduced, c_reduced)
    
    def initHidden(self, batch_size):
        if self.rnn_type == "LSTM":
            return (torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=device),
                    torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=device))
        else:
            return torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=device)

    # def init_parameters(self):
    #     for name, param in self.named_parameters():
    #         if name.startswith("embedding"): continue
    #         if param.requires_grad == True:
    #             if param.ndimension() >= 2:
    #                 nn.init.xavier_normal_(param)
    #             else:
    #                 nn.init.constant_(param, 0)

#######

class encoder_attention(nn.Module):

    def __init__(self, hidden_size=512, intra_encoder=True):
        super(encoder_attention, self).__init__()
        self.W_h = nn.Linear(512, 512, bias=False)
        self.W_s = nn.Linear(512, 512)
        self.v = nn.Linear(512, 1, bias=False)

        self.intra_encoder = intra_encoder


    def forward(self, st_hat, h, enc_padding_mask, sum_temporal_srcs):
        ''' Perform attention over encoder hidden states
        :param st_hat: decoder hidden state at current time step
        :param h: encoder hidden states
        :param enc_padding_mask:
        :param sum_temporal_srcs: if using intra-temporal attention, contains summation of attention weights from previous decoder time steps
        '''
        # Standard attention technique (eq 1 in https://arxiv.org/pdf/1704.04368.pdf)
        et = self.W_h(h)                        # bs,n_seq,2*n_hid
        dec_fea = self.W_s(st_hat).unsqueeze(1) # bs,1,2*n_hid
        et = et + dec_fea
        et = torch.tanh(et)                     # bs,n_seq,2*n_hid
        et = self.v(et).squeeze(2)              # bs,n_seq

        # intra-temporal attention     (eq 3 in https://arxiv.org/pdf/1705.04304.pdf)
        if gen_opts.intra_encoder:
            exp_et = torch.exp(et)
            if sum_temporal_srcs is None:
                et1 = exp_et
                sum_temporal_srcs  = get_cuda(torch.FloatTensor(et.size()).fill_(1e-10)) + exp_et
            else:
                et1 = exp_et/sum_temporal_srcs  #bs, n_seq
                sum_temporal_srcs = sum_temporal_srcs + exp_et
        else:
            et1 = F.softmax(et, dim=1)

        # assign 0 probability for padded elements
        at = et1 * enc_padding_mask
        normalization_factor = at.sum(1, keepdim=True)
        at = at / normalization_factor

        at = at.unsqueeze(1)                    #bs,1,n_seq
        # Compute encoder context vector
        ct_e = torch.bmm(at, h)                     #bs, 1, 2*n_hid
        ct_e = ct_e.squeeze(1)
        at = at.squeeze(1)

        return ct_e, at, sum_temporal_srcs

        #return ct_e, at, sum_temporal_srcs

class decoder_attention(nn.Module):
    def __init__(self, hidden_size = 256, intra_decoder=True):
        super(decoder_attention, self).__init__()
        self.intra_decoder = intra_decoder
        if self.intra_decoder:
            self.W_prev = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W_s = nn.Linear(hidden_size, hidden_size)
            self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, s_t, prev_s):
        '''Perform intra_decoder attention
        Args
        :param s_t: hidden state of decoder at current time step
        :param prev_s: If intra_decoder attention, contains list of previous decoder hidden states
        '''
        if self.intra_decoder is False:
            ct_d = get_cuda(torch.zeros(s_t.size()))
        elif prev_s is None:
            ct_d = get_cuda(torch.zeros(s_t.size()))
            prev_s = s_t.unsqueeze(1)               #bs, 1, n_hid
        else:
            # Standard attention technique (eq 1 in https://arxiv.org/pdf/1704.04368.pdf)
            et = self.W_prev(prev_s)                # bs,t-1,n_hid
            dec_fea = self.W_s(s_t).unsqueeze(1)    # bs,1,n_hid
            et = et + dec_fea
            et = torch.tanh(et)                         # bs,t-1,n_hid
            et = self.v(et).squeeze(2)              # bs,t-1
            # intra-decoder attention     (eq 7 & 8 in https://arxiv.org/pdf/1705.04304.pdf)
            at = F.softmax(et, dim=1).unsqueeze(1)  #bs, 1, t-1
            ct_d = torch.bmm(at, prev_s).squeeze(1)     #bs, n_hid
            prev_s = torch.cat([prev_s, s_t.unsqueeze(1)], dim=1)    #bs, t, n_hid

        return ct_d, prev_s


class Decoder(nn.Module):
    def __init__(self, hidden_size, encoder, embedding, attention=True, bias=True, dropout=0.3, fixed_embeddings=False):
        super(Decoder, self).__init__()
      # emb_dim: self.word_vec_size
      # vocab_size = self.vocab_size
        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        self.embedding.weight.data.copy_(embedding.weight)

        if fixed_embeddings:
            self.embedding.weight.requires_grad = False

        self.word_vec_size = self.embedding.embedding_dim
        self.vocab_size = self.embedding.num_embeddings

        self.hidden_size = encoder.hidden_size * encoder.num_directions
        self.num_layers = encoder.num_layers
        self.dropout = dropout
        #self.attention = attention
        
        self.enc_attention = encoder_attention() # 改为单独的文件?
        self.dec_attention = decoder_attention()
        self.x_context = nn.Linear(hidden_size*2 + self.word_vec_size, self.word_vec_size)
        # self.x_context = nn.Linear(hidden_size*2, self.word_vec_size)

        self.lstm = nn.LSTMCell(self.word_vec_size, hidden_size)
        init_lstm_wt(self.lstm)

        self.p_gen_linear = nn.Linear(hidden_size * 5 + self.word_vec_size, 1)

        #p_vocab
        self.V = nn.Linear(hidden_size*4, hidden_size)
        self.V1 = nn.Linear(hidden_size, self.vocab_size)
        init_linear_wt(self.V1)


    def forward(self, x_t, s_t, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s):
        #x_t = input_seq
        x_context_input = torch.cat([x_t, ct_e], dim=1)
        x = self.x_context(x_context_input) # context vector
        s_t = self.lstm(x, s_t) # s_t: hidden state of decoder at current time step

        dec_h, dec_c = s_t
        st_hat = torch.cat([dec_h, dec_c], dim=1)
        ct_e, attn_dist, sum_temporal_srcs = self.enc_attention(st_hat, enc_out, enc_padding_mask, sum_temporal_srcs)

        ct_d, prev_s = self.dec_attention(dec_h, prev_s)        #intra-decoder attention

        p_gen = torch.cat([ct_e, ct_d, st_hat, x], 1)
        p_gen = self.p_gen_linear(p_gen)            # bs,1
        p_gen = torch.sigmoid(p_gen)                    # bs,1

        out = torch.cat([dec_h, ct_e, ct_d], dim=1)     # bs, 4*n_hid
        out = self.V(out)                           # bs,n_hid
        out = self.V1(out)                          # bs, n_vocab
        vocab_dist = F.softmax(out, dim=1)
        vocab_dist = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_dist

        # pointer mechanism (as suggested in eq 9 https://arxiv.org/pdf/1704.04368.pdf)
        if extra_zeros is not None:
            vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=1)
        
        final_dist = vocab_dist
        for i in range(enc_batch_extend_vocab.size(0)):
            for j in range(enc_batch_extend_vocab.size(1)):
                replace_index = enc_batch_extend_vocab[i][j]
                final_dist[i][replace_index] += attn_dist_[i][j]
        # final_dist = vocab_dist.scatter_add(1, enc_batch_extend_vocab, attn_dist_) # final dist 应该为最终输出的那个 distribution

        return final_dist, s_t, ct_e, sum_temporal_srcs, prev_s

####### ENCODER #######

from helper import PAD, SOS, EOS, UNK
from gen_opts import LOAD_GEN_CHECKPOINT, gen_opts
from cuda_opts import USE_CUDA

if LOAD_GEN_CHECKPOINT:
    from gen_opts import gen_checkpoint
    from gen_dataset import gen_dataset
    encoder = Encoder(
        # embedding = nn.Embedding(gen_opts.max_vocab_size, gen_opts.embedding_dim),
                      embedding=nn.Embedding(len(gen_dataset.vocab.token2id), gen_opts.word_vec_size, padding_idx=PAD), 
                      hidden_size=gen_opts.hidden_size, 
                      num_layers=gen_opts.num_layers, 
                      dropout=gen_opts.dropout, 
                      batch_first =gen_opts.batch_first,
                      bidirectional=gen_opts.bidirectional, 
                      fixed_embeddings=gen_opts.fixed_embeddings)

    encoder.load_state_dict(gen_checkpoint['encoder_state_dict'])
else:
    #from embedding.load_emb import embedding
    from gen_dataset import gen_dataset
    encoder = Encoder(embedding=nn.Embedding(len(gen_dataset.vocab.token2id), gen_opts.word_vec_size, padding_idx=PAD), 
                      hidden_size=gen_opts.hidden_size, 
                      num_layers=gen_opts.num_layers, 
                      dropout=gen_opts.dropout, 
                      batch_first =gen_opts.batch_first,
                      bidirectional=gen_opts.bidirectional, 
                      fixed_embeddings=gen_opts.fixed_embeddings)

if USE_CUDA:
    encoder.cuda()

print(encoder)

####### DECODER #######

if LOAD_GEN_CHECKPOINT:
    from gen_opts import gen_checkpoint
    from gen_dataset import gen_dataset

    decoder = Decoder(hidden_size=gen_opts.hidden_size,
                      encoder=encoder, 
                      embedding = nn.Embedding(gen_opts.max_vocab_size, gen_opts.word_vec_size),
                      attention = gen_opts.attention, 
                      dropout=gen_opts.dropout, 
                      fixed_embeddings=gen_opts.fixed_embeddings)

    decoder.load_state_dict(gen_checkpoint['decoder_state_dict'])
    
else:
    #from embedding.load_emb import embedding #一会儿改这里
    
    decoder = Decoder(hidden_size=gen_opts.hidden_size,
                      encoder=encoder, 
                      embedding = nn.Embedding(gen_opts.max_vocab_size, gen_opts.word_vec_size), 
                      attention = gen_opts.attention, 
                      dropout=gen_opts.dropout, 
                      fixed_embeddings=gen_opts.fixed_embeddings)

if USE_CUDA:
    decoder.cuda()

print(decoder)

