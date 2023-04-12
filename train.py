# !pip install efficientnet_pytorch

import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from efficientnet_pytorch import EfficientNet
from torch import nn
from torch import optim

data_dir = '../input/petfinder-pawpularity-score/'
working_dir = './'
global_batch_size = 64
workers = 2
np.random.seed(10)

train_df = pd.read_csv('train.csv')

class PawpularityDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=transforms.ToTensor()):
        self.annotations_csv = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.annotations_csv.iloc[idx, 0])

        image = Image.open(img_name + '.jpg')
        annotations = np.array(self.annotations_csv.iloc[idx, 1:13])
        annotations = annotations.astype('float')
        score = np.array(self.annotations_csv.iloc[idx, 13])
        score = torch.tensor(score.astype('float')).view(1).to(torch.float32)

        image = self.transform(image)

        sample = [image, annotations, score]
        return sample

img_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(20),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

img_transforms_valid = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

train_data = PawpularityDataset('train.csv', 'train', transform=img_transforms)
valid_data = PawpularityDataset('train.csv', 'train', transform=img_transforms_valid)

np.random.seed(13)
valid_size = 0.1
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(train_data, batch_size=global_batch_size,
                          sampler=train_sampler, num_workers=workers,
                          pin_memory=True) 
valid_loader = DataLoader(valid_data, batch_size=global_batch_size,
                          sampler=valid_sampler, num_workers=workers,
                          pin_memory=True) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'efficientnet-b0'
model = EfficientNet.from_pretrained(model_name)

num_features = model._fc.in_features
model._fc = nn.Linear(num_features, 1)

model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, _, scores in train_loader:
        images, scores = images.to(device), scores.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, scores)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    scheduler.step()

    running_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}")

    # Evaluate the model on the validation set
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, _, scores in valid_loader:
            images, scores = images.to(device), scores.to(device)

            outputs = model(images)
            loss = criterion(outputs, scores)

            running_val_loss += loss.item() * images.size(0)

    running_val_loss /= len(valid_loader.dataset)
    print(f"Validation Loss: {running_val_loss:.4f}")

model_save_path = 'trained_efficientnet_b0.pth'
torch.save(model.state_dict(), model_save_path)

print(f"Model saved to: {model_save_path}")