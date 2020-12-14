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
    gen_opts.hidden_size = 128 # encoder embedding 定义的是 256
    gen_opts.num_layers = 1 # 一开始是 2
    gen_opts.dropout = 0.3
    gen_opts.bidirectional = True
    gen_opts.attention = True
    gen_opts.fixed_embeddings = False

    gen_opts.batch_first = True

    # Configure optimization
    gen_opts.learning_rate = 0.0001 #0.001
    gen_opts.lr_decay_steps = 100 #40 #8
    gen_opts.lr_decay_rate = 1 #0.992556794 #1.02734584 #1.007704814136 #0.988553095 #0.91201084 #0.9623506264 #0.981855
    
    # Configure training
    gen_opts.max_seq_len = 120 # max sequence length to prevent OOM.
    gen_opts.batch_size = 32 # 256 # 20 # 64
    gen_opts.num_epochs = 3
    gen_opts.print_every_step = 4 # 100 # 20
    gen_opts.save_every_step = 8 # 100 # 20

    
    gen_opts.max_iterations = 500000 # 最大 iter 数

    gen_opts.max_decoding_steps = 25 # TS 说 99% 都小于 15

    # gen_opts.embedding_dim=256
    
    # Configure vocabulary size
    gen_opts.filter_vocab = True       # Vocab size too large may cause the problem of CUDA error: out of memory
    gen_opts.max_vocab_size = 50000    # work only if filter_vocab is True

    gen_opts.intra_encoder = True