import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from config import MOVEMENTS_PER_SESSION
import numpy as np
import torch.nn as nn
import torchvision
import time
from torchvision import datasets, models, transforms
from PIL import Image



def get_label_from_filename(filepath):
    """Given the filepath of .h5 data, return the correspondent label

    E.g.
    S13_session2_mov2_7500events.h5 -> Session 2, movement 2 -> label 10
    """

    label = 0
    filename = os.path.basename(filepath)
    session = int(filename[filename.find('session_') + len('session_')])
    mov = int(filename[filename.find('mov_') + len('mov_')])

    for i in range(1, session):
        label += MOVEMENTS_PER_SESSION[i]

    return label + mov - 1


class DHP19Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, indexes=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sampletot += n_frames.
        """

        self.x_paths = sorted(glob.glob(os.path.join(root_dir, "*.npy")))
        self.x_indexes = indexes if indexes is not None else np.arange(
            len(self.x_paths))
        self.labels = [get_label_from_filename(
            x_path) for x_path in self.x_paths]

        self.transform = transform

    def __len__(self):
        return len(self.x_indexes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.x_paths[idx]
        x = np.load(img_name)
        x = np.repeat(x[:, :,  np.newaxis], 3, axis=-1)
        x = Image.fromarray(x, 'RGB')
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


class CNN(nn.Module):
    def __init__(self, input_shape=(260, 346), n_classes=33):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 250, 346)
            nn.Conv2d(
                in_channels=1,              # input depth
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                # if want same width and length of this image after onv2d, padding=(kernel_size-1)/2 if stride=
                padding=2,
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            # choose max value in 2x2 area, output shape (16, 14, 14)
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(         # input shape (16, 130, 173)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 65, 86)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 65, 86)
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 65 * 86, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    LR = 0.001
    EPOCH = 10
    NUM_CLASSES = 33

    cnn = models.resnet18(pretrained=False)

    num_ftrs = cnn.fc.in_features
    cnn.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    cnn.cuda()
    print(cnn)  # net architecture

 # /home/gianscarpe/dev/data/h5_dataset_7500_events/movements_per_frame
    root_dir = '/home/gianscarpe/dev/data/h5_dataset_7500_events/movements_per_frame'
    n_frames = len(os.listdir(root_dir))
    indexes = np.arange(n_frames)
    np.random.shuffle(indexes)

    train_index = indexes[:int(.8 * n_frames)]
    val_index = indexes[int(.8 * n_frames):]

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))])

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])

    train_loader = DataLoader(DHP19Dataset(
        root_dir, train_index, transform=preprocess), batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(DHP19Dataset(
        root_dir, val_index, transform=preprocess), batch_size=32, shuffle=True, num_workers=4)

    # optimize all cnn parameters
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    # the target label is not one-hotted
    loss_func = nn.CrossEntropyLoss()
    n_train_data = len(train_loader)
    n_val_data = len(val_loader)

    STAMP_EVERY = 100
    for epoch in range(EPOCH):

        start_time = time.time()
        accumulated_loss = 0
        # gives batch data, normalize x when iterate train_loader
        for step, (b_x, b_y) in enumerate(train_loader):

            b_x = b_x.float()

            output = cnn(b_x.cuda())               # cnn output
            loss = loss_func(output, b_y.cuda())   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # appnly gradients

            accumulated_loss += loss.cpu().data.numpy()
            if (step % STAMP_EVERY == 0):
                print(
                    f"Epoch: {epoch} - [{step/n_train_data*100:.2f}%] | train loss: {accumulated_loss/(step+1):.4f}")

        accuracy = 0
        for step, (b_x, b_y) in enumerate(val_loader):
            b_x = b_x.float()
            test_output = cnn(b_x.cuda())

            pred_y = np.argmax(test_output.cpu().data.numpy(), 1)

            accuracy += float((pred_y == b_y.data.numpy()).astype(int)
                              .sum())

        print(
            f"Epoch: {epoch} | Duration: {(start_time - time.time()):.4f}s | val accuracy: {(accuracy/n_val_data):.4f}")
