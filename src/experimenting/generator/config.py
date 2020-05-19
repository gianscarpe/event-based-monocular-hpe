BASE_DIR = "/home/gianscarpe/dev/event-camera"
import torch

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
