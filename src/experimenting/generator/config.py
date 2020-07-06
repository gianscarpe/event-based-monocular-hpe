import torch
BASE_DIR = "/home/gscarpellini/dev/event-camera/src/experimenting"

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
