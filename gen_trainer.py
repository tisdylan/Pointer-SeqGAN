import torch
import torch.nn.functional as F
import torch.nn as nn
# from masked_cross_entropy import masked_cross_entropy
from helper import PAD, SOS, EOS, UNK
from gen_opts import gen_opts
from torch.autograd import Variable
# import time

# from rouge import Rouge

unk_id = UNK
pad_id = PAD
start_id = SOS
end_id = EOS

eps = 1e-12

def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def gen_trainer(encoder, decoder, encoder_optim, decoder_optim, max_dec_steps, dec_lens,
                src_seqs, tgt_seqs, src_lens, tgt_lens, enc_lens, extend_vocab_size, target_batch, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, USE_CUDA=True): 
    # time_start=time.time()
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Last batch might not have the same size as we set to the `batch_size`

    batch_size = src_seqs.size(1)
    assert(batch_size == tgt_seqs.size(0))
    
    # Pack tensors to variables for neural network inputs (in order to autograd)

    src_lens = torch.LongTensor(src_lens)
    # tgt_lens = torch.LongTensor(tgt_lens) # 已经是 LongTensor 了

    # Decoder's input
    # input_seq = torch.LongTensor([SOS] * batch_size)
    
    # Decoder's output sequence length = max target sequence length of current batch.
    ## Changed to mean. Otherwise it always results in out-of-memory
    # tgt_lens_float = tgt_lens.float()
    # max_tgt_len = int((int(tgt_lens_float.mean() * batch_size) - int(tgt_lens_float.min()) - int(tgt_lens_float.max()))/(batch_size-2) -15 )
    # del tgt_lens_float
    # print("Decoder Steps: ", max_tgt_len)
    
    # Store all decoder's outputs.
    # **CRUTIAL** 
    # Don't set:
    # >> decoder_outputs = Variable(torch.zeros(max_tgt_len, batch_size, decoder.vocab_size))
    # Varying tensor size could cause GPU allocate a new memory causing OOM, 
    # so we intialize tensor with fixed size instead:
    # `gen_opts.max_seq_len` is a fixed number, unlike `max_tgt_len` always varys.
    # decoder_outputs = torch.zeros(gen_opts.max_seq_len, batch_size, decoder.vocab_size) # 实际上设置为定长（2000）才会 OOM
    decoder_outputs = Variable(torch.zeros(max_dec_steps, batch_size, extend_vocab_size))

    # Move variables from CPU to GPU.
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        # tgt_seqs = tgt_seqs.cuda()
        src_lens = src_lens.cuda()
        # tgt_lens = tgt_lens.cuda()
        # enc_lens = enc_lens.cuda()
        # extend_vocab_size = extend_vocab_size.cuda()
        enc_padding_mask = enc_padding_mask.cuda()
        ct_e = ct_e.cuda()
        extra_zeros = extra_zeros.cuda()
        enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        # input_seq = input_seq.cuda()
        decoder_outputs = decoder_outputs.cuda()
        target_batch = target_batch.cuda()
        

    # -------------------------------------
    # Training mode (enable dropout)
    # -------------------------------------
    encoder.train()
    decoder.train()
    
    # -------------------------------------
    # Zero gradients, since optimizers will accumulate gradients for every backward.
    # -------------------------------------
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    # dataset 中 src_seqs 生成的时候加入了 extend vocab，可以在这里删掉超纲的词。
    # print("Embedding...")
    # x = encoder.embedding(src_seqs) # 转到 encoder 内部了

    # print("Input into encoder...")
    # encoder_outputs, encoder_hidden = encoder(x, enc_lens)
    # del x
    
    # print(src_seqs.cpu().numpy().tolist())

    encoder_outputs, encoder_hidden = encoder(src_seqs, enc_lens)

    # -------------------------------------
    # Forward decoder
    # -------------------------------------
    # Initialize decoder's hidden state as encoder's last hidden state.
    
    # rouge = Rouge()

    step_losses = []
    s_t = (encoder_hidden[0], encoder_hidden[1]) # Decoder hidden states
    x_t = get_cuda(torch.LongTensor(len(encoder_outputs)).fill_(start_id)) # Input to the decoder

    prev_s = None  # Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
    sum_temporal_srcs = None  # Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
    
    # inds = [] ## for rouge
    
    # Run through decoder one time step at a time.
    # for t in range(min(max_dec_len, gen_opts.max_decoding_steps)):
    for t in range(max_dec_steps):
        # print("Decoder Steps: ", max_dec_steps)
        
        use_gound_truth = get_cuda((torch.rand(len(encoder_outputs)) > 0.25)).long() #Probabilities indicating whether to use ground truth labels instead of previous decoded tokens
        x_t = use_gound_truth * target_batch[:, t] + (1 - use_gound_truth) * x_t #Select decoder input based on use_ground_truth probabilities
        is_oov = (x_t >= gen_opts.max_vocab_size).long()  #Mask indicating whether sampled word is OOV
        x_t = (1 - is_oov) * x_t.detach() + (is_oov) * unk_id  #Replace OOVs with [UNK] token
        
        x_t = decoder.embedding(x_t)

        decoder_output, s_t, ct_e, sum_temporal_srcs, prev_s \
                = decoder(x_t, s_t, encoder_outputs, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s) # 去掉了第四项 src_lens
            
        decoder_outputs[t] = decoder_output
        target = target_batch[:, t]
        log_probs = get_cuda(torch.log(decoder_output + eps))
        step_loss = get_cuda(F.nll_loss(log_probs, target, reduction="none", ignore_index=pad_id))
        step_losses.append(step_loss)

        x_t = torch.multinomial(decoder_output, 1).squeeze() #Sample words from final distribution which can be used as input in next time step
        # ## for rouge
        # _, x_t_rouge = torch.max(decoder_output, dim=1)
        # x_t_rouge = x_t_rouge.detach()
        # inds.append(x_t_rouge)
        # print(x_t.cpu().numpy().tolist())
        # print(target.cpu().numpy().tolist())
        
        del decoder_output
    # print(inds)
    # -------------------------------------
    # Compute loss
    # -------------------------------------
    # decoder_outputs = decoder_outputs[:max_tgt_len].transpose(0,1).contiguous()
    # tgt_seqs = tgt_seqs.transpose(0,1).contiguous()
    torch.cuda.empty_cache()
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
    # print("\nComputing loss...")
    losses = get_cuda(torch.sum(torch.stack(step_losses, 1), 1))#unnormalized losses for each example in the batch; (batch_size)
    batch_avg_loss = losses / dec_lens          #Normalized losses; (batch_size)
    loss = get_cuda(torch.mean(batch_avg_loss))
    # print("Loss:", str(loss))
    
    # -------------------------------------
    # Backward and optimize
    # -------------------------------------
    # Backward to get gradients w.r.t parameters in model.
    loss.backward()
    # print("\nGot gradient.")
    # time_end2=time.time()
    # print('Loss backward time cost: ',time_end2-time_end1,'s')
    # Update parameters with optimizers
    encoder_optim.step()
    decoder_optim.step()
    # print("\nModel updated.")
    torch.cuda.empty_cache()
    
    num_words = tgt_lens.sum().item()

    del src_seqs, tgt_seqs, src_lens, tgt_lens, extend_vocab_size, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, s_t, sum_temporal_srcs, prev_s, encoder_outputs, encoder_hidden, target_batch, target, step_losses, log_probs, step_loss
    torch.cuda.empty_cache()

    return loss.item(), num_words