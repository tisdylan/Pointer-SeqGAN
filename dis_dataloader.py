import torch
from torch.utils.data import DataLoader
import random

from gen_dataloader import get_gen_iter
from dis_dataset import DisDataset
from dis_collate_fn import dis_collate_fn
from dis_opts import dis_opts

from gen_dataset import get_gen_dataset
from generator import generator

def get_dis_iter(dataset = None, training_pairs=None, num_data=None, num_workers=0):
    
    if training_pairs is not None:
        from gen_dataset import gen_dataset
        gen_iter = get_gen_iter(gen_dataset=gen_dataset, batch_size=20)

        if num_data:
            dis_dataset = DisDataset(dataset = dataset, training_pairs = random.sample(training_pairs, num_data), gen_iter = gen_iter, generator = generator)
            # dis_dataset = DisDataset(random.sample(training_pairs, num_data))
        else:
            # dis_dataset = DisDataset(training_pairs = training_pairs)
            dis_dataset = DisDataset(dataset = dataset, training_pairs = training_pairs, gen_iter = gen_iter, generator = generator)
            # dataset = None, training_pairs = None, gen_iter = None, generator = None
            # dataset = dataset, training_pairs = None, num_data=None, num_workers=0
            # dataset = dataset, training_pairs = None, gen_iter = gen_iter, generator = generator

    else:
        if num_data:
            gen_dataset = get_gen_dataset(num_data=num_data/2)
        else:
            from gen_dataset import gen_dataset

        # 64 is the batch_size which could work best
        gen_iter = get_gen_iter(gen_dataset=gen_dataset,
                                batch_size=20)
        
        dis_dataset = DisDataset(dataset = dataset, training_pairs = None, gen_iter = gen_iter, generator = generator)
    
    
    dis_iter = DataLoader(dataset=dis_dataset,
                          batch_size=dis_opts.batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          collate_fn=dis_collate_fn)

    return dis_iter