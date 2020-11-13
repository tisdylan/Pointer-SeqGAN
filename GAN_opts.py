from AttrDict import AttrDict

GAN_opts = AttrDict()

# Configure adversarial training

GAN_opts.G_print_every_step = 1
GAN_opts.G_save_every_step = 25

GAN_opts.D_print_every_step = 25
GAN_opts.D_save_every_step = 25

GAN_opts.dis_num_epoch = 2
GAN_opts.batch_size = 20 # 原来是 64
GAN_opts.d_step_repeat_times = 25

GAN_opts.num_rollout = 4

#GAN_opts.pretrain_data_path= "repository/chunked/pretrain/train_*"
#GAN_opts.train_data_path = 	"repository/chunked/train/train_*"
#GAN_opts.valid_data_path = 	"repository/chunked/valid/valid_*"
#GAN_opts.test_data_path = 	"repository/chunked/test/test_*"
#GAN_opts.vocab_path = 		"/usr/local/Convolutional SeqGAN/vocab"
