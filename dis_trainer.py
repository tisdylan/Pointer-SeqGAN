import torch
import torch.nn as nn

from discriminator import discriminator
from dis_optimizer import dis_optim

criterion = nn.NLLLoss(reduction='sum')


def dis_trainer(doc_seqs, summary_seqs, labels, discriminator, dis_optim, USE_CUDA=True):
    """             
        - doc_seqs:       (batch_size, max_seq_len>=dis_opts.conv_padding_len)
        - summary_seqs:    (batch_size, max_seq_len>=dis_opts.conv_padding_len)
        - labels:           (batch,)
    """
    batch_size = doc_seqs.size(0)
    assert(batch_size == summary_seqs.size(0))
    assert(batch_size == labels.size(0))

    # Training mode (enable dropout)
    discriminator.train()

    if USE_CUDA:
        doc_seqs = doc_seqs.cuda()
        summary_seqs = summary_seqs.cuda()
        labels = labels.cuda()

    # (batch_size, 2)
    # doc_seqs:    (batch, max_seq_len)
    # summary_seqs: (batch, max_seq_len)
    pred = discriminator(doc_seqs, summary_seqs)

    loss = criterion(pred, labels)
    num_corrects = (pred.max(1)[1] == labels).sum().item()

    dis_optim.zero_grad()
    loss.backward()
    dis_optim.step()


    del doc_seqs, summary_seqs, labels, pred
    torch.cuda.empty_cache()

    return loss.item(), num_corrects