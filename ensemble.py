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
from pathlib import Path
get_ipython().run_line_magic('env', 'TOKENIZERS_PARALLELISM=false')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[2]:


class CFG1:
    print_freq = 3000
    num_workers = 4
    uns_model = "/root/使用tree进行处理的代码/paraphrase-multilingual-mpnet-base-v2-exp_fold0_epochs10_description"
    sup_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    uns_tokenizer = AutoTokenizer.from_pretrained(uns_model)
    sup_tokenizer = AutoTokenizer.from_pretrained(sup_model)
    gradient_checkpointing = False
    batch_size = 32
    n_folds = 5
    top_n = 33
    seed = 42
    threshold = 0.02


# In[3]:


class CFG2:
    print_freq = 3000
    num_workers = 4
    uns_model = "/root/使用title训练出的模型/stage-1-paraphrase-multilingual-mpnet-base-v2-tuned-4747"
    sup_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    uns_tokenizer = AutoTokenizer.from_pretrained(uns_model)
    sup_tokenizer = AutoTokenizer.from_pretrained(sup_model)
    gradient_checkpointing = False
    batch_size = 32
    n_folds = 5
    top_n = 33
    seed = 42
    threshold = 0.02


# In[4]:


def read_data1(cfg,to):
    topics = to
    content = pd.read_csv('/root/content.csv')
    sample_submission = pd.read_csv('/root/sample_submission.csv')
    # Merge topics with sample submission to only infer test topics
    topics = topics.merge(sample_submission, how = 'inner', left_on = 'id', right_on = 'topic_id')
    topics['title']=topics['title']+' '+topics['tree']
    content['title']=content['title']
    # Fillna titles
    topics['title'].fillna("", inplace = True)
    content['title'].fillna("", inplace = True)
    # Sort by title length to make inference faster
    topics['length'] = topics['title'].apply(lambda x: len(x))
    content['length'] = content['title'].apply(lambda x: len(x))
    topics.sort_values('length', inplace = True)
    content.sort_values('length', inplace = True)
    # Drop cols
    topics.drop(['description', 'channel', 'category', 'level', 'language', 'parent', 'has_content', 'length', 'topic_id', 'content_ids','tree'], axis = 1, inplace = True)
    content.drop(['description', 'kind', 'language', 'text', 'copyright_holder', 'license', 'length'], axis = 1, inplace = True)
    # Reset index
    topics.reset_index(drop = True, inplace = True)
    content.reset_index(drop = True, inplace = True)
    print(' ')
    print('-' * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")
    return topics, content


# In[5]:


def read_data2(cfg,to):
    topics = to
    content = pd.read_csv('/root/content.csv')
    sample_submission = pd.read_csv('/root/sample_submission.csv')
    # Merge topics with sample submission to only infer test topics
    topics = topics.merge(sample_submission, how = 'inner', left_on = 'id', right_on = 'topic_id')
    topics['title']=topics['title']
    content['title']=content['title']
    # Fillna titles
    topics['title'].fillna("", inplace = True)
    content['title'].fillna("", inplace = True)
    # Sort by title length to make inference faster
    topics['length'] = topics['title'].apply(lambda x: len(x))
    content['length'] = content['title'].apply(lambda x: len(x))
    topics.sort_values('length', inplace = True)
    content.sort_values('length', inplace = True)
    # Drop cols
    topics.drop(['description', 'channel', 'category', 'level', 'language', 'parent', 'has_content', 'length', 'topic_id', 'content_ids','tree'], axis = 1, inplace = True)
    content.drop(['description', 'kind', 'language', 'text', 'copyright_holder', 'license', 'length'], axis = 1, inplace = True)
    # Reset index
    topics.reset_index(drop = True, inplace = True)
    content.reset_index(drop = True, inplace = True)
    print(' ')
    print('-' * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")
    return topics, content


# In[6]:


def prepare_uns_input(text, cfg):
    inputs = cfg.uns_tokenizer.encode_plus(
        text, 
        return_tensors = None, 
        add_special_tokens = True, 
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs


# In[7]:


class uns_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['title'].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_uns_input(self.texts[item], self.cfg)
        return inputs


# In[8]:


def prepare_sup_input(text, cfg):
    inputs = cfg.sup_tokenizer.encode_plus(
        text, 
        return_tensors = None, 
        add_special_tokens = True, 
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs


# In[9]:


class sup_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['text'].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_sup_input(self.texts[item], self.cfg)
        return inputs


# In[10]:


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


# In[11]:


class uns_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.uns_model)
        self.model = AutoModel.from_pretrained(cfg.uns_model, config = self.config)
        self.pool = MeanPooling()
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature
    def forward(self, inputs):
        feature = self.feature(inputs)
        return feature


# In[12]:


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


# In[13]:


def get_pos_socre(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
    return round(np.mean(int_true), 5)


# In[14]:


def build_inference_set(topics, content, cfg):
    # Create lists for training
    topics_ids = []
    content_ids = []
    title1 = []
    title2 = []
    # Iterate over each topic
    for k in tqdm(range(len(topics))):
        row = topics.iloc[k]
        topics_id = row['id']
        topics_title = row['title']
        predictions = row['predictions'].split(' ')
        for pred in predictions:
            content_title = content.loc[pred, 'title']
            topics_ids.append(topics_id)
            content_ids.append(pred)
            title1.append(topics_title)
            title2.append(content_title)
    # Build training dataset
    test = pd.DataFrame(
        {'topics_ids': topics_ids, 
         'content_ids': content_ids, 
         'title1': title1, 
         'title2': title2
        }
    )
    # Release memory
    del topics_ids, content_ids, title1, title2
    gc.collect()
    return test


# In[15]:


class custom_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.sup_model, output_hidden_states = True)
        self.config.hidden_dropout = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.config.attention_dropout = 0.0
        self.config.attention_probs_dropout_prob = 0.0
        self.model = AutoModel.from_pretrained(cfg.sup_model, config = self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature
    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output


# In[16]:


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
        collate_fn = DataCollatorWithPadding(tokenizer = cfg.uns_tokenizer, padding = 'longest'),
        num_workers = cfg.num_workers, 
        pin_memory = True, 
        drop_last = False
    )
    content_loader = DataLoader(
        content_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = cfg.uns_tokenizer, padding = 'longest'),
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


# In[17]:


def preprocess_test(test):
    test['title1'].fillna("Title does not exist", inplace = True)
    test['title2'].fillna("Title does not exist", inplace = True)
    # Create feature column
    test['text'] = test['title1'] + '[SEP]' + test['title2']
    # Drop titles
    test.drop(['title1', 'title2'], axis = 1, inplace = True)
    # Sort so inference is faster
    test['length'] = test['text'].apply(lambda x: len(x))
    test.sort_values('length', inplace = True)
    test.drop(['length'], axis = 1, inplace = True)
    test.reset_index(drop = True, inplace = True)
    gc.collect()
    return test


# In[18]:


class custom_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.sup_model, output_hidden_states = True)
        self.config.hidden_dropout = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.config.attention_dropout = 0.0
        self.config.attention_probs_dropout_prob = 0.0
        self.model = AutoModel.from_pretrained(cfg.sup_model, config = self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature
    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output


# In[19]:


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total = len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().squeeze().to('cpu').numpy().reshape(-1))
    predictions = np.concatenate(preds)
    return predictions


# In[20]:


topics_test=pd.read_csv('/root/topics_tree.csv')


# In[21]:


topics_tree,content_tree=read_data1(CFG1,topics_test)
topics_title,content_title=read_data2(CFG2,topics_test)


# In[22]:


topics_tree,content_tree=get_neighbors(topics_tree,content_tree,CFG1)
topics_title,content_title=get_neighbors(topics_title,content_title,CFG2) 
gc.collect()


# In[23]:


content_tree.set_index('id', inplace = True)
# Build training set
test_tree = build_inference_set(topics_tree, content_tree, CFG1)
# Process test set
test_tree = preprocess_test(test_tree)
content_title.set_index('id', inplace = True)
# Build training set
test_title = build_inference_set(topics_title, content_title, CFG2)
# Process test set
test_title = preprocess_test(test_title)


# In[41]:


def inference(test1, test2, cfg1, cfg2):
    # Create dataset and loader
    test_dataset_tree = sup_dataset(test1, cfg1)
    test_dataset_title = sup_dataset(test2, cfg2)
    test_loader_tree = DataLoader(
        test_dataset_tree, 
        batch_size = cfg1.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = cfg1.sup_tokenizer, padding = 'longest'),
        num_workers = cfg1.num_workers, 
        pin_memory = True, 
        drop_last = False
    )
    test_loader_title = DataLoader(
        test_dataset_title, 
        batch_size = cfg2.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = cfg2.sup_tokenizer, padding = 'longest'),
        num_workers = cfg2.num_workers, 
        pin_memory = True, 
        drop_last = False
    )
    # Get model
    model_tree_path=[
            '/root/使用tree进行处理的代码/-root-使用tree进行处理的代码-paraphrase-multilingual-mpnet-base-v2-exp_fold0_epochs10_description_fold0_42.pth',
    ]  
    model_title_path=[
        '/root/使用title训练出的模型/0.495.pth',
    ]
    pre_tree=[]
    pre_title=[]
    for model_path in model_tree_path:
        model=custom_model(cfg1)
        state = torch.load(model_path, map_location = torch.device('cpu'))
        model.load_state_dict(state['model'])
        pre_tree.append(inference_fn(test_loader_tree,model,device))
    for model_path in model_title_path:
        model=custom_model(cfg2)
        state = torch.load(model_path, map_location = torch.device('cpu'))
        model.load_state_dict(state['model'])
        pre_title.append(inference_fn(test_loader_title,model,device))
    pre_tree=np.average(pre_tree,axis=0)
    pre_title=np.average(pre_title,axis=0)
    prediction=(pre_tree+pre_title)/2
    torch.cuda.empty_cache()
    del test_dataset_tree.test_dataser_title,test_loader_tree,test_loader_title,model_tree,model_title,state
    gc.collect
    # Use threshold
    test['prediction'] = prediction
    test['predictions_binary'] = (prediction>CFG.threshold).astype(int)
    test = test.sort_values('prediction', ascending = False)
    predicted = test[test.predictions_binary == 1].groupby(['topics_ids'])['content_ids'].agg(list).reset_index()
    no_pos = test.groupby(['topics_ids']).head(1)
    no_pos = no_pos[no_pos.predictions_binary==0].groupby(['topics_ids'])['content_ids'].agg(list).reset_index()
    predicted = pd.concat([predicted,no_pos]).reset_index(drop=True)
    predicted['content_ids'] = predicted['content_ids'].apply(lambda x: ' '.join(x))
    predicted.columns = ['topic_id', 'content_ids']
    predicted.to_csv('submission.csv', index = False)
    return predicted


# In[42]:


test_r = inference(test_tree,test_title,CFG1,CFG2)
test_r.head()


# In[ ]:




