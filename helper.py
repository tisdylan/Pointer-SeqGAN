import os

BASE_DIR = os.path.dirname("./")
REPO_DIR = os.path.join(BASE_DIR, "repo")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
GEN_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "generator")
DIS_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "discriminator")

PAD = 0
SOS = 1
EOS = 2
UNK = 3