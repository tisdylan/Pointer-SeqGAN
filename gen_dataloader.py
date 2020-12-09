import torch
from torch.utils.data import DataLoader

from gen_dataset import gen_dataset
from gen_collate_fn import gen_collate_fn
from gen_opts import gen_opts

def get_gen_iter(gen_dataset, batch_size, num_workers=0, pin_memory=True):
    gen_iter = DataLoader(dataset=gen_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          collate_fn=gen_collate_fn,
                          pin_memory=pin_memory)
    return gen_iter