#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import time
import math
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
import cupy as cp
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from sentence_transformers import SentenceTransformer

get_ipython().run_line_magic('env', 'TOKENIZERS_PARALLELISM=false')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[2]:


class CFG:
    num_workers = 4
    model = '/root/autodl-tmp/paraphrase-multilingual-mpnet-base-v2-exp'
    tokenizer = AutoTokenizer.from_pretrained(model)
    batch_size = 32
    top_n = 50
    seed = 42
    


# In[3]:


def read_data(cfg):
    topics = pd.read_csv('/root/autodl-tmp/topics.csv')
    content = pd.read_csv('/root/autodl-tmp/content.csv')
    correlations = pd.read_csv('/root//kfold_correlations.csv')
    correlations = correlations[correlations.fold == 0]
    # Fillna titles
    topics['title'].fillna("", inplace = True)
    content['title'].fillna("", inplace = True)
    # Fillna descriptions
    topics['description'].fillna("", inplace = True)
    content['description'].fillna("", inplace = True)
    #将title和description进行结合，单纯使用title和description进行处理
    topics['title']=topics['title']+topics['description']
    content['title']=content['title']+content['description']+topics['tree']
    # Sort by title length to make inference faster
    topics['length'] = topics['title'].apply(lambda x: len(x))
    content['length'] = content['title'].apply(lambda x: len(x))
    topics.sort_values('length', inplace = True)
    content.sort_values('length', inplace = True)
    # Drop cols
    topics.drop(['description', 'channel', 'category', 'level', 'language', 'parent', 'has_content', 'length'], axis = 1, inplace = True)
    content.drop(['description', 'kind', 'language', 'text', 'copyright_holder', 'license', 'length'], axis = 1, inplace = True)
    # Reset index
    topics.reset_index(drop = True, inplace = True)
    content.reset_index(drop = True, inplace = True)
    print(' ')
    print('-' * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")
    print(f"correlations.shape: {correlations.shape}")
    return topics, content, correlations


# In[4]:


def prepare_input(text, cfg):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        max_length = 64,
        truncation=True,
        return_tensors = None, 
        add_special_tokens = True, 
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs


# In[5]:


class uns_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['title'].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.cfg)
        return inputs


# In[6]:


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


# In[7]:


class uns_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model)
        self.model = AutoModel.from_pretrained(cfg.model, config = self.config)
        self.pool = MeanPooling()
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature
    def forward(self, inputs):
        feature = self.feature(inputs)
        return feature


# In[8]:


def get_embeddings(loader, model, device):
    model.eval()
    preds = []
    for step, inputs in enumerate(tqdm(loader)):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    preds = np.concatenate(preds)
    return preds


# In[9]:


def get_pos_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
    return round(np.mean(int_true), 5)


# In[10]:


def f2_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    return round(f2.mean(), 4)


# In[11]:


def build_training_set(topics, content, cfg):
    # Create lists for training
    topics_ids = []
    content_ids = []
    title1 = []
    title2 = []
    targets = []
    folds = []
    # Iterate over each topic
    for k in tqdm(range(len(topics))):
        row = topics.iloc[k]
        topics_id = row['id']
        topics_title = row['title']
        predictions = row['predictions'].split(' ')
        ground_truth = row['content_ids'].split(' ')
        fold = row['fold']
        for pred in predictions:
            content_title = content.loc[pred, 'title']
            topics_ids.append(topics_id)
            content_ids.append(pred)
            title1.append(topics_title)
            title2.append(content_title)
            folds.append(fold)
            # If pred is in ground truth, 1 else 0
            if pred in ground_truth:
                targets.append(1)
            else:
                targets.append(0)
    # Build training dataset
    train = pd.DataFrame(
        {'topics_ids': topics_ids, 
         'content_ids': content_ids, 
         'title1': title1, 
         'title2': title2, 
         'target': targets,
         'fold' : folds}
    )
    # Release memory
    del topics_ids, content_ids, title1, title2, targets
    gc.collect()
    return train


# In[12]:


def get_neighbors(topics, content, cfg):
    # Create topics dataset
    topics_dataset = uns_dataset(topics, cfg)
    # Create content dataset
    content_dataset = uns_dataset(content, cfg)
    #这里将topics和content返回为张量
    #Create topics and content dataloaders
    topics_loader = DataLoader(
        topics_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = cfg.tokenizer, padding = 'longest'),
        num_workers = cfg.num_workers, 
        pin_memory = True, 
        drop_last = False
    )
    content_loader = DataLoader(
        content_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = cfg.tokenizer, padding = 'longest'),
        num_workers = cfg.num_workers, 
        pin_memory = True, 
        drop_last = False
        )
    # Create unsupervised model to extract embeddings
    model = uns_model(cfg)
    model.to(device)#模型加载到相应设备中
    # Predict topics
    topics_preds = get_embeddings(topics_loader, model, device)#
    content_preds = get_embeddings(content_loader, model, device)#这里进行嵌入
    # Transfer predictions to gpu
    topics_preds_gpu1 = np.array(topics_preds)
    content_preds_gpu1 = np.array(content_preds)
    topics_preds_gpu=np.mat(topics_preds_gpu1)
    content_preds_gpu=np.mat(content_preds_gpu1)
    # Release memory
    torch.cuda.empty_cache()
    del topics_dataset, content_dataset, topics_loader, content_loader, topics_preds, content_preds
    gc.collect()
    # KNN model
    print(' ')
    print('Training KNN model...')
    neighbors_model = NearestNeighbors(n_neighbors = cfg.top_n, metric = 'cosine')
    neighbors_model.fit(content_preds_gpu)
    indices1 = neighbors_model.kneighbors(topics_preds_gpu, return_distance = False)
    indices=cp.array(indices1)
    predictions = []
    for k in tqdm(range(len(indices))):
        pred = indices[k]
        p = ' '.join([content.loc[ind, 'id'] for ind in pred.get()])
        predictions.append(p)
    topics['predictions'] = predictions
    # Release memory
    del topics_preds_gpu, content_preds_gpu, neighbors_model, predictions, indices, model
    gc.collect()
    return topics, content


# In[13]:


topics, content, correlations = read_data(CFG)


# In[14]:


topics, content = get_neighbors(topics, content, CFG)


# In[15]:


topics_test = topics.merge(correlations, how = 'inner', left_on = ['id'], right_on = ['topic_id'])
pos_score = get_pos_score(topics_test['content_ids'], topics_test['predictions'])
print(f'Our max positive score is {pos_score}')


# In[16]:


f_score = f2_score(topics_test['content_ids'], topics_test['predictions'])
print(f'Our f2_score is {f_score}')


# In[17]:


del correlations
gc.collect()


# In[18]:


content.set_index('id', inplace = True)


# In[19]:


full_correlations = pd.read_csv('/root/kfold_correlations.csv')
topics_full = topics.merge(full_correlations, how = 'inner', left_on = ['id'], right_on = ['topic_id'])
topics_full['predictions'] = topics_full.apply(lambda x: ' '.join(list(set(x.predictions.split(' ') + x.content_ids.split(' ')))) \
                                               if x.fold != 0 else x.predictions, axis = 1)
train = build_training_set(topics_full, content, CFG)
print(f'Our training set has {len(train)} rows')
# Save train set to disk to train on another notebook
train.to_csv('train_top30_fold0_cv_with_groundtruth_final.csv', index = False)
train.head()


# In[ ]:




