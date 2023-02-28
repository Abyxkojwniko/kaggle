#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold


# In[10]:


DATA_PATH = "/root/"
topics = pd.read_csv(DATA_PATH + "使用tree进行处理的代码/topics_tree.csv")
content = pd.read_csv(DATA_PATH + "content.csv")
correlations = pd.read_csv(DATA_PATH + "correlations.csv")
correlations.shape


# In[11]:


def cv_split(train, n_folds, seed):
    kfold = KFold(n_splits = n_folds, shuffle = True, random_state = seed)
    for num, (train_index, val_index) in enumerate(kfold.split(train)):
        train.loc[val_index, 'fold'] = int(num)
    train['fold'] = train['fold'].astype(int)
    return train


# In[12]:


kfolds = cv_split(correlations, 5, 42)
correlations = kfolds[kfolds.fold != 0]
correlations


# In[13]:


kfold_correlations=kfolds[kfolds.fold == 0]
kfold_correlations.to_csv('kfold_correlations.csv')
kfold_correlations


# In[14]:


topics.rename(columns=lambda x: "topic_" + x, inplace=True)
content.rename(columns=lambda x: "content_" + x, inplace=True)


# In[15]:


correlations["content_id"] = correlations["content_ids"].str.split(" ")
corr = correlations.explode("content_id").drop(columns=["content_ids"])


# In[16]:


corr = corr.merge(topics, how="left", on="topic_id")
corr = corr.merge(content, how="left", on="content_id")
corr.head()


# In[17]:


#将title和description进行结合，单纯使用title和description进行处理
corr['topic_title']=corr['topic_title']+corr['topic_description']
corr['content_title']=corr['content_title']+corr['content_description']+corr['topic_tree']


# In[18]:


corr


# In[19]:


corr["set"] = corr[["topic_title", "content_title"]].values.tolist()
train_df = pd.DataFrame(corr["set"])


# In[20]:


dataset = Dataset.from_pandas(train_df)


# In[ ]:


train_examples = []
train_data = dataset["set"]
n_examples = dataset.num_rows

for i in range(n_examples):
    example = train_data[i]
    if example[0] == None: #remove None
        print(example)
        continue        
    train_examples.append(InputExample(texts=[str(example[0]), str(example[1])]))


# In[ ]:


model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


# In[ ]:


train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
train_loss = losses.MultipleNegativesRankingLoss(model=model)
num_epochs = 15
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data


# In[ ]:


model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          save_best_model = True,
          output_path='./paraphrase-multilingual-mpnet-base-v2-exp_fold0_epochs10',
          warmup_steps=warmup_steps)


# In[ ]:




