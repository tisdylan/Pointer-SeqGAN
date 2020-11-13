import torch
import torch.nn.functional as F

from cuda_opts import USE_CUDA
from rollout import rollout

def gan_loss(src_seqs, response_seqs, src_lens, response_lens, decoder_outputs, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s, extend_vocab_size, num_rollout=16):

    """
        - src_seqs:         (seq_len, batch_size)
        - src_lens:         [1] * batch_size

        - response_seqs:    (batch_size, max_out_len)
        - response_lens:    (batch_size)

        - decoder_outputs:  (batch_size, max_response_len, vocab_size)

        # src_seqs, out_seqs, src_lens, out_lens, decoder_outputs,
    """

    # - rewards: (batch_size, max_response_len)
    # Use torch.no_grad() is very important, or it would cause CUDA OOM error very easily.
    with torch.no_grad():
        print("\nDoing rollout...")
        rewards = rollout.get_reward(src_seqs, response_seqs, src_lens, response_lens, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s, extend_vocab_size, num_rollout=num_rollout)

    # - decoder_outputs:  (batch_size, max_response_len, vocab_size)
    decoder_outputs = F.log_softmax(decoder_outputs, dim=2)
    
    # (batch_size, max_response_len) * (batch_size, max_response_len) = (batch_size, max_response_len)
    loss = decoder_outputs.topk(1)[0].squeeze(2) * rewards

    loss = - loss.sum()

    del rewards, decoder_outputs
    torch.cuda.empty_cache()

    return loss