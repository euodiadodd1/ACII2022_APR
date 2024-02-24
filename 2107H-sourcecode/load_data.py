import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
from PIL import Image
from glob import glob



class FeatureFusionDataset(Dataset):
  def __init__(self, data_filenames, label_filenames, p_list):
    self.data_filenames = data_filenames
    self.label_filenames = label_filenames
    self.p_list = p_list
    self.transform = transforms.Compose([transforms.ToTensor()])

  def __len__(self):
    return len(self.data_filenames)

  def __getitem__(self, idx):
    gaze_data = pd.read_csv(self.data_filenames[idx][0], header=None, 
                 index_col=False)
    au_data = pd.read_csv(self.data_filenames[idx][1], header=None, 
                 index_col=False)
    pose_data = pd.read_csv(self.data_filenames[idx][2], header=None, 
                 index_col=False)
    labels = self.label_filenames[idx].astype(float)
    p = self.p_list[idx]

    data = pd.concat([pose_data,gaze_data, au_data], axis = 0).dropna().to_numpy().astype(float)
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    scaled_data = scaler.fit_transform(data)
    
    if idx == self.__len__():  
            raise IndexError  

    return self.transform(scaled_data), labels,p

class DecisionFusionDataset(Dataset):
  def __init__(self, data_filenames, label_filenames, p_list):
    self.data_filenames = data_filenames
    self.label_filenames = label_filenames
    self.p_list = p_list
    self.transform = transforms.Compose([transforms.ToTensor()])

  def __len__(self):
    return len(self.data_filenames)

  def __getitem__(self, idx):

    p = self.p_list[idx]
    gaze_data = pd.read_csv(self.data_filenames[idx][0], header=None, 
                 index_col=False, skiprows=1)
    au_data = pd.read_csv(self.data_filenames[idx][1], header=None, 
                 index_col=False, skiprows=1)
    pose_data = pd.read_csv(self.data_filenames[idx][2], header=None, 
                 index_col=False, skiprows=1)
    labels = self.label_filenames[idx].astype(float)

    data = [pose_data,gaze_data, au_data]
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    data = [self.transform(scaler.fit_transform(x.to_numpy().astype(float))) for x in data]

    if idx == self.__len__():  
            raise IndexError  

    return data, labels, p


class DyadicFeatureFusionDataset(Dataset):
    def __init__(self, data_filenames, label_filenames, p_list):
      self.data_filenames = data_filenames
      self.label_filenames = label_filenames
      self.p_list = p_list
      self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
      return len(self.data_filenames)

    def __getitem__(self, idx):
      p = self.p_list[idx]
      p1 = np.concatenate([pd.read_csv(i, header=None, 
                 index_col=False) for i in self.data_filenames[idx][0]])
      p2 = np.concatenate([pd.read_csv(i, header=None, 
                 index_col=False) for i in self.data_filenames[idx][1]])

      labels = self.label_filenames[idx].astype(float)

      data = np.concatenate([p1,p2]).astype(float)
      scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
      data = scaler.fit_transform(data)
      
      if idx == self.__len__():  
              raise IndexError  
  
      return self.transform(data), labels, p

class DyadicDecisionFusionDataset(Dataset):
    def __init__(self, data_filenames, label_filenames, p_list):
      self.data_filenames = data_filenames
      self.label_filenames = label_filenames
      self.p_list = p_list
      self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
      return len(self.data_filenames)

    def __getitem__(self, idx):
      p = self.p_list[idx]
      p1 = [pd.read_csv(i, header=None, 
                 index_col=False) for i in self.data_filenames[idx][0]]
      p2 = [pd.read_csv(i, header=None, 
                 index_col=False) for i in self.data_filenames[idx][1]]

      labels = self.label_filenames[idx].astype(float)

      au1,gaze1,pose1,aud1 =  p1
      au2,gaze2,pose2,aud2 =  p2
      
      data = [au1,gaze1,pose1,aud1,au2,gaze2,pose2,aud2]
      scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
      data = [self.transform(scaler.fit_transform(x).astype(float)) for x in data]
      
      if idx == self.__len__():  
              raise IndexError  
      #print(d.shape,l.shape)
      return data, labels, p

class IdvDyadicDecisionFusionDataset(Dataset):
    def __init__(self, data_filenames, label_filenames, p_list):
      self.data_filenames = data_filenames
      self.label_filenames = label_filenames
      self.p_list = p_list
      self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
      return len(self.data_filenames)

    def __getitem__(self, idx):
      p = self.p_list[idx]
      p1 = [pd.read_csv(i, header=None, 
                 index_col=False) for i in self.data_filenames[idx][0]]
      p2 = [pd.read_csv(i, header=None, 
                 index_col=False) for i in self.data_filenames[idx][1]]

      p1_vis = np.concatenate(p1[1:])
      p1_aud = p1[0].to_numpy()
      p2_vis = np.concatenate(p2[1:])
      p2_aud = p2[0].to_numpy()

      labels = np.concatenate(self.label_filenames[idx]).astype(float)

      data = [p1_vis,p1_aud,p2_vis,p2_aud]
      data = [self.transform(x.astype(float)) for x in data]
      
      if idx == self.__len__():  
              raise IndexError  

      return data, labels, p


## Fetch labels for each session's participants
def get_labels(part, dir):
  if part == "train":
    df = pd.read_csv("metadata_train/parts_train.csv")
    df2 = pd.read_csv("metadata_train/sessions_train.csv")
  elif part == "val":
    df = pd.read_csv("metadata_val/parts_val_unmasked.csv")
    df2 = pd.read_csv("metadata_val/sessions_val.csv")
  else:
    df = pd.read_csv("metadata_test/parts_test_unmasked.csv")
    df2 = pd.read_csv("metadata_test/sessions_test.csv")
        

  label_idx = ["OPENMINDEDNESS_Z", "CONSCIENTIOUSNESS_Z", "EXTRAVERSION_Z", "AGREEABLENESS_Z", "NEGATIVEEMOTIONALITY_Z"]
  participants = df2.loc[df2["ID"] == dir]
  p1 = participants["PART.1"].values[0]
  p2 = participants["PART.2"].values[0]
  l1 = np.asarray(df.loc[df["ID"].isin([p1,p2])][label_idx].values)
  return p1,p2, l1

def fused_data(fusion_strategy, batch_size, section, dyad=False, split="val"):

    ## download and load training dataset

    pose_dir_list = np.sort([x for x in glob(section +"_spectral/pose/*")])
    gaze_dir_list = np.sort([x for x in glob(section +"_spectral/gaze/*")])
    au_dir_list = np.sort([x for x in glob(section +"_spectral/au/*")])
    aud_dir_list = np.sort([x for x in glob(section +"_audio_spec_train/*")])

    train_dir_list = np.column_stack([pose_dir_list,gaze_dir_list,au_dir_list,aud_dir_list])

    ts_pose_dir_list = np.sort([x for x in glob(section +"_spectral_"+split+"/pose/*")])
    ts_gaze_dir_list = np.sort([x for x in glob(section +"_spectral_"+split+"/gaze/*")])
    ts_au_dir_list = np.sort([x for x in glob(section +"_spectral_"+split+"/au/*")])
    ts_aud_dir_list = np.sort([x for x in glob(section +"_audio_spec_"+split+"/*")])

    test_dir_list = np.column_stack([ts_pose_dir_list,ts_gaze_dir_list, ts_au_dir_list, ts_aud_dir_list])

    train_dirs = train_dir_list
    test_dirs = test_dir_list
    
    train_file_list = np.zeros((1,4))
    train_labels = np.zeros((1,5))
    test_file_list = np.zeros((1,4))
    test_labels =  np.zeros((1,5))
    train_p = []
    test_p = []



    for i in train_dirs:
        dir = int(os.path.split(i[0])[1])
        gaze = np.sort([x for x in glob(i[1]+"/*.csv")])
        pose = np.sort([x for x in glob(i[0]+"/*.csv")])
        au = np.sort([x for x in glob(i[2]+"/*.csv")])
        aud = np.sort([x for x in glob(i[3]+"/*.csv")])
        

        ls = np.column_stack([pose,gaze,au,aud])
        p1,p2,labels = get_labels("train",dir)
        train_file_list = np.concatenate([train_file_list, ls])
        train_labels = np.vstack([train_labels, labels[0], labels[1]])
        train_p += [p1,p2]
        
    for i in test_dirs:
        dir = int(int(os.path.split(i[0])[1]))
        gaze = np.sort([x for x in glob(i[1]+"/*.csv")])
        pose = np.sort([x for x in glob(i[0]+"/*.csv")])
        au = np.sort([x for x in glob(i[2]+"/*.csv")])
        aud = np.sort([x for x in glob(i[3]+"/*.csv")])

        ls = np.column_stack([pose,gaze,au,aud])
        p1,p2, labels = get_labels(split, dir)
  
        test_file_list = np.concatenate([test_file_list, ls])
        test_labels = np.vstack([test_labels, labels[0], labels[1]])
        test_p += [p1]
        test_p += [p2]

    train_file_list = np.delete(train_file_list, 0, axis=0)
    test_file_list = np.delete(test_file_list, 0, axis=0)
    train_labels = np.delete(train_labels, 0, axis=0)
    test_labels = np.delete(test_labels, 0,  axis=0)



    dims = [pd.read_csv(glob(i+"/*.csv")[0], header=None, 
                 index_col=False).shape for i in train_dirs[0]]

    train_file_list.sort()
    test_file_list.sort()
    train_labels.sort()
    test_labels.sort()

    if dyad:
      print("Dyadic")
      train_file_list = np.repeat(np.split(train_file_list, len(train_file_list)//2), 2, axis=0)
      test_file_list = np.repeat(np.split(test_file_list, len(test_file_list)//2), 2, axis=0)
      #p_list = np.repeat(p_list,2)
      
      if fusion_strategy in ["avg_decision","decision","attention"]:
        dims =  [(4,80),(18,80),(72,80),(14,80),(4,80),(72,80),(18,80),(14,80)]
        train_dataset = DyadicDecisionFusionDataset(data_filenames = train_file_list,label_filenames= train_labels, p_list = train_p)
        test_dataset = DyadicDecisionFusionDataset(data_filenames = test_file_list, label_filenames= test_labels, p_list = test_p)

      else: 
          train_dataset = DyadicFeatureFusionDataset(data_filenames = train_file_list,label_filenames= train_labels, p_list = train_p)
          test_dataset = DyadicFeatureFusionDataset(data_filenames = test_file_list, label_filenames= test_labels, p_list = test_p)

    else:
      if fusion_strategy in ["avg_decision","decision","attention"]:

          train_dataset = DecisionFusionDataset(data_filenames = train_file_list,label_filenames= train_labels, p_list = train_p)
          test_dataset = DecisionFusionDataset(data_filenames = test_file_list, label_filenames= test_labels, p_list = test_p)

      else: 
          train_dataset = FeatureFusionDataset(data_filenames = train_file_list,label_filenames= train_labels, p_list = train_p)
          test_dataset = FeatureFusionDataset(data_filenames = test_file_list, label_filenames= test_labels, p_list = test_p)


    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=0)


    return trainloader, testloader, dims

