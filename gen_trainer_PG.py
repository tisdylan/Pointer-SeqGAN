import torch
import torch.nn as nn
from gan_loss import gan_loss

def gen_trainer_PG(src_seqs, src_lens, generator, encoder_optim, decoder_optim, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s, extend_vocab_size, num_rollout=16, USE_CUDA=True):
    """
        - src_seqs:     Tensor,     shape: (seq_len, batch_size)
        - src_lens:     list,       shape: [1] * batch_size
    """    

    # - out_seqs:           (batch_size, max_out_len)
    # - out_lens:           (batch_size)
    # - decoder_outputs:    (batch_size, max_out_len, vocab_size)
    out_seqs, out_lens, decoder_outputs = generator(src_seqs, src_lens, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s, extend_vocab_size, enable_dropout=True, state=None)

    #out_seqs, out_lens, decoder_outputs = generator(src_seqs, src_lens, enable_dropout=True)
    # print("\nGot generator output. Computing loss...")

    loss = gan_loss(src_seqs, out_seqs, src_lens, out_lens, decoder_outputs, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s, extend_vocab_size, num_rollout=num_rollout)
    
    #print("\nGot loss.")

    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    loss.backward()
    
    # nn.utils.clip_grad_value_(generator.parameters(), clip_value=1.1)
    nn.utils.clip_grad_value_(generator.encoder.parameters(), clip_value=1.1)
    nn.utils.clip_grad_value_(generator.decoder.parameters(), clip_value=1.1)
    # nn.utils.clip_grad_norm_(generator.encoder.rnn.parameters(), max_norm=20, norm_type=2)
    # nn.utils.clip_grad_norm_(generator.decoder.lstm.parameters(), max_norm=20, norm_type=2)
    
    encoder_optim.step()
    decoder_optim.step()

    num_words = out_lens.sum().item()

    del out_seqs, out_lens, decoder_outputs, src_seqs, src_lens, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s, extend_vocab_size
    
    torch.cuda.empty_cache()

    return loss.item(), num_words