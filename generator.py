import torch
import torch.nn as nn
import torch.nn.functional as F

from gen_opts import gen_opts
from helper import PAD, SOS, EOS, UNK
from sequence_mask import sequence_mask
from torch.autograd import Variable

unk_id = UNK
pad_id = PAD
start_id = SOS
end_id = EOS

def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

class Generator(nn.Module):
    def __init__(self, encoder, decoder, USE_CUDA=True):
        super(Generator, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.USE_CUDA = USE_CUDA

    def forward(self, src_seqs, src_lens, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s, extend_vocab_size, enable_dropout=True, state=None):
        """
            - src_seqs: Tensor, shape: (seq_len, batch_size)
            - src_lens:  list,  shape: [1] * batch_size

        :param enc_padding_mask: Mask for encoder input; Tensor of size (batch_size, length_input_sequence) with values of 0 for pad tokens & 1 for others
        :param ct_e: encoder context vector for time_step=0 (eq 5 in https://arxiv.org/pdf/1705.04304.pdf)
        :param extra_zeros: Tensor used to extend vocab distribution for pointer mechanism
        :param enc_batch_extend_vocab: Input batch that stores OOV ids
        """

        batch_size = src_seqs.size(1)
        src_lens = torch.LongTensor(src_lens)   # (batch_size)

        #input_seq = torch.LongTensor([SOS] * batch_size)
        # gen_opts.max_seq_len
        out_seqs = torch.zeros(gen_opts.max_decoding_steps, batch_size).long()
        out_lens = torch.zeros(batch_size).long()
        # decoder_outputs = torch.zeros(gen_opts.max_seq_len, batch_size, self.decoder.vocab_size)
        decoder_outputs = Variable(torch.zeros(gen_opts.max_decoding_steps, batch_size, extend_vocab_size))

        prev_s = None # Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        sum_temporal_srcs = None # Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)

        if self.USE_CUDA:
            src_seqs = src_seqs.cuda()
            src_lens = src_lens.cuda()
            #input_seq = input_seq.cuda()

        if enable_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        # - encoder_outputs: (max_seq_len, batch_size, hidden_size)
        # - encoder_hidden:  (num_layers, batch_size, num_directions * hidden_size)
        encoder_outputs, encoder_hidden = self.encoder(src_seqs, src_lens)

        s_t = (encoder_hidden[0], encoder_hidden[1])  # Decoder hidden states
        #decoder_hidden = encoder_hidden # 是不是重复了??
        
        x_t = get_cuda(torch.LongTensor(len(encoder_outputs)).fill_(SOS)) # Input to the decoder

        state_len = 0

        if state is not None:
            """
                state is used to do the monte carlo search

                - state: (state_len, batch_size)
            """
            state_len = state.size(0)
            
            for t in range(state_len):
                is_oov = (x_t >= gen_opts.max_vocab_size).long()  #Mask indicating whether sampled word is OOV
                x_t = (1 - is_oov) * x_t.detach() + (is_oov) * unk_id  #Replace OOVs with [UNK] token
                # - decoder_output   : (batch_size, vocab_size)
                # - s_t: decoder_hidden   : (num_layers, batch_size, hidden_size)
                # - attention_weights: (batch_size, max_src_len)
                x_t = decoder.embedding(x_t)
                
                decoder_output, s_t, ct_e, sum_temporal_srcs, prev_s \
                = self.decoder(x_t, s_t, encoder_outputs, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s) # 去掉了第四项 src_lens # 本来输入是 input_seq
                # x_t, s_t, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s
                
                decoder_outputs[t] = decoder_output

                out_seqs[t] = state[t]  # (batch_size)

                if (state[t] == EOS).sum().item() != 0:     # someone has finished generating seqs
                    out_lens += ((out_lens == 0) * (state[t] == EOS) * (t+1)).long()

                # input_seq = state[t]    # (batch_size)
                # if self.USE_CUDA:
                #     input_seq = input_seq.cuda()
                x_t = state[t]

                is_oov = (x_t >= gen_opts.max_vocab_size).long()  #Mask indicating whether sampled word is OOV
                x_t = (1 - is_oov) * x_t.detach() + (is_oov) * unk_id  #Replace OOVs with [UNK] token
                # x_t = torch.multinomial(decoder_output, 1).squeeze() #Sample words from final distribution which can be used as input in next time step
                # is_oov = (x_t >= gen_opts.max_vocab_size).long()  #Mask indicating whether sampled word is OOV
                # x_t = (1 - is_oov) * x_t.detach() + (is_oov) * unk_id  #Replace OOVs with [UNK] token


        for t in range(state_len, gen_opts.max_decoding_steps):
            
            is_oov = (x_t >= gen_opts.max_vocab_size).long()  #Mask indicating whether sampled word is OOV
            x_t = (1 - is_oov) * x_t.detach() + (is_oov) * unk_id  #Replace OOVs with [UNK] token

            #x_t = decoder.embedding(x_t)
            x_t = decoder.embedding(x_t.long())

            # - decoder_output   : (batch_size, vocab_size)
            # - s_t: decoder_hidden   : (num_layers, batch_size, hidden_size)
            # - attention_weights: (batch_size, max_src_len)
            decoder_output, s_t, ct_e, sum_temporal_srcs, prev_s \
            = self.decoder(x_t, s_t, encoder_outputs, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s) # 去掉了第四项 src_lens

            decoder_outputs[t] = decoder_output


            if state is not None:
                # Sample a word from the given probability distribution
                prob = F.softmax(decoder_output, dim=1)     # (batch_size, vocab_size)
                token_id = prob.multinomial(1)              # (batch_size, 1)
            else:
                # Choose top word from decoder's output
                # - prob:       (batch_size, 1)
                # - token_id:   (batch_size, 1)
                prob, token_id = decoder_output.topk(1)

            # (batch_size, 1) -> (batch_size)
            token_id = token_id.squeeze(1)
            
            out_seqs[t] = token_id

            if (token_id == EOS).sum().item() != 0:     # someone has finished generating seqs
                out_lens += ((out_lens == 0) * (token_id == EOS).cpu() * (t+1)).long()

            if (out_lens == 0).sum().item() == 0:       # everyone has finished generating seqs
                break
            
            #input_seq = token_id
            x_t = token_id

            is_oov = (x_t >= gen_opts.max_vocab_size).long()  #Mask indicating whether sampled word is OOV
            x_t = (1 - is_oov) * x_t.detach() + (is_oov) * unk_id  #Replace OOVs with [UNK] token

        # if someone doesn't have lens, that means, it ended the for-loop without meeting <EOS>
        #gen_opts.max_seq_len
        out_lens += ((out_lens == 0) * gen_opts.max_decoding_steps).long()     

        max_out_len = out_lens.max().item()
        
        # (max_out_len, batch_size, vocab_size) -> (batch_size, max_out_len, vocab_size)
        decoder_outputs = decoder_outputs[:max_out_len].transpose(0,1)

        # (max_out_len, batch_size) -> (batch_size, max_out_len)
        out_seqs = out_seqs[:max_out_len].transpose(0,1)
        mask = sequence_mask(out_lens)                      # (batch_size, max_out_len)
        #mask = mask.t()
        out_seqs.masked_fill_(~mask, PAD)                # (batch_size, max_out_len)

        #del src_seqs, src_lens, input_seq, mask, batch_size, encoder_outputs, encoder_hidden, decoder_output, s_t, state_len, prob, token_id, max_out_len
        
        torch.cuda.empty_cache()
        
        """
            - out_seqs:           (batch_size, max_out_len)
            - out_lens:           (batch_size)
            - decoder_outputs:    (batch_size, max_out_len, vocab_size)
        """
        return out_seqs, out_lens, decoder_outputs


## -----------------------------------------------------------------------------------

from cuda_opts import USE_CUDA
from encoder_decoder import encoder
from encoder_decoder import decoder

generator = Generator(encoder=encoder,
                      decoder=decoder,
                      USE_CUDA=USE_CUDA)