import torch

from helper import EOS
from dis_opts import dis_opts
from cuda_opts import USE_CUDA
from sequence_mask import sequence_mask


class Rollout(object):
    def __init__(self, generator, discriminator):
    
        self.generator = generator
        self.discriminator = discriminator

    def get_reward(self, src_seqs, summary_seqs, src_lens, summary_lens, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s, extend_vocab_size, num_rollout=16):
        """
            - src_seqs:         (seq_len, batch_size)
            - src_lens:         [1] * batch_size

            - summary_seqs:    (batch_size, max_out_len)
            - summary_lens:    (batch_size)
        """
        
        max_summary_len = summary_seqs.size(1)
        batch_size = src_seqs.size(1)

        rewards = torch.zeros(batch_size, max_summary_len)     # (batch_size, max_summary_len)
        
        doc_seqs = self.pad_seqs(src_seqs.t())
        if USE_CUDA:
            doc_seqs = doc_seqs.cuda()

        # - summary_seqs: (max_out_len, batch_size)
        summary_seqs = summary_seqs.t()
        
        for i in range(num_rollout):
            for l in range(1, max_summary_len):
                state = summary_seqs[:l]   # (state_len, batch_size)

                """
                    - out_seqs:           (batch_size, max_out_len)
                    - out_lens:           (batch_size)
                    - decoder_outputs:    (batch_size, max_out_len, vocab_size)
                """
                # out_seqs, out_lens, decoder_outputs = self.generator(src_seqs, src_lens, enable_dropout=True, state=state)

                out_seqs, out_lens, decoder_outputs = self.generator(src_seqs, src_lens, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s, extend_vocab_size, enable_dropout = True, state = state)
            

                """
                    - doc_seqs:     (batch_size, doc_max_seq_len>=20)
                    - summary_seqs:  (batch_size, summary_max_seq_len>=20)
                    - reward:         (batch_size, 1)
                """
                
                out_seqs = self.pad_seqs(out_seqs)
                if USE_CUDA:
                    out_seqs = out_seqs.cuda()

                reward = - self.discriminator(doc_seqs, out_seqs)[:, 1]

                rewards[:, l-1] += reward.cpu()
                
                del out_seqs, out_lens, decoder_outputs, reward
                torch.cuda.empty_cache()


            # for the last token, there's no need to do the summary again
            out_seqs = self.pad_seqs(summary_seqs.t())
            if USE_CUDA:
                out_seqs = out_seqs.cuda()

            reward = - self.discriminator(doc_seqs, out_seqs)[:, 1]
            
            rewards[:, max_summary_len-1] += reward.cpu()
            
            del out_seqs, reward
            torch.cuda.empty_cache()


        mask = sequence_mask(summary_lens).float()   # (batch_size, max_summary_len)
        rewards = rewards * mask                      # (batch_size, max_summary_len)
        rewards = rewards / num_rollout               # (batch_size, max_summary_len)

        del doc_seqs, mask
        torch.cuda.empty_cache()

        return rewards


    def pad_seqs(self, seqs):
        # - seqs: (batch_size, max_out_len)

        seqs_len = seqs.size(1)
        if seqs_len  < dis_opts.conv_padding_len:    
            padded_seqs = torch.zeros(seqs.size(0), dis_opts.conv_padding_len).long()
            for i, seq in enumerate(seqs):
                padded_seqs[i, :seqs_len] = seq
            return padded_seqs

        else:
            return seqs


## -----------------------------------------------------------------------------------

from generator import generator
from discriminator import discriminator

rollout = Rollout(generator, discriminator)