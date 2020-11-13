import torch

""" Enable GPU training """

USE_CUDA = torch.cuda.is_available()
print('Use_CUDA={}'.format(USE_CUDA))

device = torch.device("cuda" if USE_CUDA else "cpu")

if USE_CUDA:
    # You can change device by `torch.cuda.set_device(device_id)`
    print("device_count={}".format(torch.cuda.device_count()))
    print('current_device={}'.format(torch.cuda.current_device()))

# torch.cuda.set_device(0)