from torch.utils.data import Dataset, DataLoader
import os
import random
import torch 
import numpy as np
import pickle
from torch.autograd import Variable
import cv2
import torchvision.models as models
import torch.nn as nn


class DataGenerator(Dataset):
	
	def __init__(self, x, y):
		self.files = self.get_files(x, y)
        
	def __len__(self):
		return len(self.files)
        

	def __getitem__(self,idx):

		src_path, tgt = self.files[idx]
		src = cv2.imread(src_path)
		src = cv2.resize(src, (224, 224))
		src = np.array(src)
		src = np.transpose(src,(2, 0, 1))
		return src, tgt
			
	def get_files(self,x, y):

		data = []
		for i in range(len(x)):
			data.append((x[i],y[i]))
		return data

	
def collate_fn_customised(data):

	source = []
	tgt = []
	target = []

	for d in data:
		source.append(d[0])
		tgt.append(d[1])
	source = np.array(source,dtype=np.float32)
	labels = list(set(tgt))
	labels.sort()

	id2label={i: c for i, c in enumerate(labels)},
	label2id={c: i for i, c in enumerate(labels)}

	for t in tgt:
		target.append(label2id[t])
	target = np.array(target)

	return torch.tensor(source), torch.tensor(target)

def load_data(x, y, batch_size=128, num_workers=2, shuffle=True):
    
	dataset = DataGenerator(x, y)
	data_loader = DataLoader(dataset, collate_fn = collate_fn_customised, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

	return data_loader

#data_path = "./data/multi30k/uncompressed_data"
#load_data(data_path)