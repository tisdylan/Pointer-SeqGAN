import torch.optim as optim

from encoder_decoder import encoder
from encoder_decoder import decoder
from gen_opts import LOAD_GEN_CHECKPOINT, gen_opts

encoder_optim = optim.Adam([p for p in encoder.parameters() if p.requires_grad], lr=gen_opts.learning_rate)
decoder_optim = optim.Adam([p for p in decoder.parameters() if p.requires_grad], lr=gen_opts.learning_rate)

encoder_optim_scheduler = torch.optim.lr_scheduler.ExponentialLR(encoder_optim, gamma=0.941205) # do after 50 steps
decoder_optim_scheduler = torch.optim.lr_scheduler.ExponentialLR(decoder_optim, gamma=0.941205) # do after 50 steps

if LOAD_GEN_CHECKPOINT:
    from gen_opts import gen_checkpoint
    encoder_optim.load_state_dict(gen_checkpoint["encoder_optim_state_dict"])
    decoder_optim.load_state_dict(gen_checkpoint["decoder_optim_state_dict"])