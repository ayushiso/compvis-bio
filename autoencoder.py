import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import random
import numpy as np
import cv2
from sklearn.cluster import KMeans 

        
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True   


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.enc_conv_1 = nn.Conv2d(3,6, kernel_size=8)
        self.enc_max_1 = nn.MaxPool2d(4, 4, return_indices=True)
        self.enc_conv_2 = nn.Conv2d(6, 16, kernel_size=8)
        self.enc_max_2 = nn.MaxPool2d(4, 4, return_indices=True)

        self.dec_conv_2 = nn.ConvTranspose2d(16, 6, kernel_size=8)
        self.dec_max_2 = nn.MaxUnpool2d(4,4)
        self.dec_conv_1 = nn.ConvTranspose2d(6, 3, kernel_size=8)
        self.dec_max_1 = nn.MaxUnpool2d(4,4)

    def forward(self, x):
        # encoding part
        
        x = self.enc_conv_1(x)
        x = F.relu(x)
        size1 = x.size()
        x, indices_1 = self.enc_max_1(x)
        x = self.enc_conv_2(x)
        x = F.relu(x)
        size2 = x.size()
        x, indices_2 = self.enc_max_2(x)
        
        
        # decoding part
        fx = self.dec_max_2(x, indices_2, output_size=size2)
        fx = self.dec_conv_2(F.relu(fx))
        fx = self.dec_max_1(fx, indices_1, output_size=size1)
        fx = self.dec_conv_1(F.relu(fx))
        fx = torch.tanh(fx)
        
        return x, fx

      
class myDataset(Dataset):
  def __init__(self, data_root):
    self.samples = []
    
    for organelle in os.listdir(data_root):
      org_folder = os.path.join(data_root, organelle)
      
      for pic in os.listdir(org_folder):
        pic_path = os.path.join(org_folder, pic)
        
        img = cv2.imread(pic_path)
        
        self.samples.append((img, organelle))
        

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    transform = transforms.Compose([transforms.ToTensor()])
    
    tens = transform(self.samples[idx][0])
    
    return tens, self.samples[idx][1]


def train(model, device, iterator, optimizer, criterion):
    
    epoch_loss = 0  
    for data in iterator:
      
        x, _ = data        
        x = x.to(device)
        
        optimizer.zero_grad()
                
        _, fx = model(x)
            
        loss = criterion(fx, x)

        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)  


def show(img, fname):
    npimg = img.detach().cpu().numpy()
    npimg = np.clip(npimg, 0, 1, out=npimg)
    plt.imsave(fname, np.transpose(npimg, (1,2,0)))
  
def img_save(model, device, train_iterator):
    
    visited_organelles = {}
  
    for (x, y) in train_iterator:
      
      x = x.to(device)
      _, fx = model(x)

      for (act, y_val, pred) in zip(x, y, fx):
        
        if y_val not in visited_organelles:
          f = y_val + ".png"
          visited_organelles[y] = 1      
          mylist = [act, pred]
          show(make_grid(mylist), f)


def max_count(clusters):
  max_count = 0
  counts = {}
  for number in clusters:
    if number not in counts.keys():
      counts[number] = 1
    else: 
      counts[number] += 1
      
  for val in counts.values():
    if max_count < val:
      max_count = val
      
  return max_count
    
          
          
def purity_score(y_true, y_pred):

  # collect different groups by predicted cluster labels
  cluster_groups = {}
  
  for i in range(len(y_pred)):
    if y_pred[i] not in cluster_groups.keys():
      cluster_groups[y_pred[i]] = [y_true[i]]
    else:
      cluster_groups[y_pred[i]].append(y_true[i])
  
  # get max from each cluster group and return mean
  running_max = 0.0
  for cluster in cluster_groups.values():
    # print(cluster)
    running_max += max_count(cluster)
    
  return running_max / len(y_true)


if __name__ == "__main__":
  
    dataset = myDataset('./Hela/train')
    train_iterator = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    print("finished loading data")        
                        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Autoencoder()
    print("initialized model")
    model.cuda()

    optimizer = optim.Adam(model.parameters())

    criterion = nn.MSELoss()


    ### Training ###
    EPOCHS = 10
      
    for epoch in range(EPOCHS):
        train_loss = train(model, device, train_iterator, optimizer, criterion)
        print("Epoch:", epoch+1, "| loss:", train_loss)


    # save images
    img_save(model, device, train_iterator)
        
    # get latent encoding from trained model
    encodings = []
    classes = []
    
    for (x, y) in train_iterator:
      
      x = x.to(device)
      x_enc, _ = model(x)
      x_enc = x_enc.view(x_enc.size()[0], 9744)
      encodings.append(x_enc)
      classes.append(list(y))
      
    encodings = torch.cat(encodings, dim=0)
    encodings = encodings.cpu().detach().numpy()

    batch_classes = []
    for batch in classes:
      batch_classes.append(np.array(batch))
    classes = np.concatenate(batch_classes, axis = 0)
      
    clf = KMeans(n_clusters= 10)
    clf.fit(encodings)
    cluster_labels = clf.predict(encodings)
    
    print(purity_score(classes, cluster_labels))