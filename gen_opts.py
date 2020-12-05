import os
import torch

from helper import GEN_CHECKPOINT_DIR

from AttrDict import AttrDict
from checkpoint import load_checkpoint

# If enabled, load checkpoint.
LOAD_GEN_CHECKPOINT = False

if LOAD_GEN_CHECKPOINT:
    # Modify this path.
    gen_checkpoint_path = os.path.join(GEN_CHECKPOINT_DIR, "<Your filename>")
    gen_checkpoint = load_checkpoint(gen_checkpoint_path)
    gen_opts = gen_checkpoint['opts']

else:
    gen_opts = AttrDict()

    # Configure models
    gen_opts.word_vec_size = 256 # 一开始写了 300...
    #gen_opts.rnn_type = 'LSTM'
    gen_opts.hidden_size = 256 # encoder embedding 定义的是 256
    gen_opts.num_layers = 1 # 一开始是 2
    gen_opts.dropout = 0.3
    gen_opts.bidirectional = True
    gen_opts.attention = True
    gen_opts.fixed_embeddings = False

    gen_opts.batch_first = True

    # Configure optimization
    gen_opts.learning_rate = 0.0001 #0.001
    
    # Configure training
    gen_opts.max_seq_len = 120          # max sequence length to prevent OOM.
    gen_opts.batch_size = 20
    gen_opts.num_epochs = 3
    gen_opts.print_every_step = 100
    gen_opts.save_every_step = 100
    # gen_opts.num_epochs = 1
    # gen_opts.print_every_step = 30
    # gen_opts.save_every_step = 50
    gen_opts.lr_decay_steps = 50 # decay lr after each 50 steps (0.941205)
    
    gen_opts.max_iterations = 500000 # 最大 iter 数

    gen_opts.max_decoding_steps = 25 # TS 说 99% 都小于 15

    # gen_opts.embedding_dim=256
    
    # Configure vocabulary size
    gen_opts.filter_vocab = True       # Vocab size too large may cause the problem of CUDA error: out of memory
    gen_opts.max_vocab_size = 5000    # work only if filter_vocab is True

    gen_opts.intra_encoder = True