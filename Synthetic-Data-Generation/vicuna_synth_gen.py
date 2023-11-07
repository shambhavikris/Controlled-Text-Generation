#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install jsonlines


# In[2]:


import os
import transformers
import torch
import time
import pandas as pd
import numpy as np
import jsonlines # useful
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer


# In[3]:


# restart kernel after installation
# this is the only thing needed, everything else is already installed on Kaggle.
# !pip install bitsandbytes 


# In[4]:


model_id = "hiepnh/Wizard-Vicuna-7B-Uncensored-HF-sharded"
device = torch.device("cuda")


# In[5]:


def get_model_and_tokenizer(model_id):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'left'
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


# In[6]:


model, tokenizer = get_model_and_tokenizer(model_id)


# In[7]:


def generate_text(model, tokenizer, prompt, prefix, max_new_tokens=75, temperature=2.5, num_return_sequences = 3):
    model_inputs = tokenizer(prompt + prefix, return_tensors='pt').to(device)
    model_output = model.generate(**model_inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                top_k=10,
                                temperature=temperature,
                                num_return_sequences=num_return_sequences)
    output_text = tokenizer.batch_decode(model_output, skip_special_tokens=True)
    return output_text


# ### Some sample generations (instructing the model to be positive)

# In[8]:


# %%time
# prompt = "User: Generate a very uplifting, inspiring, positive, continuation for the statement:\n\n"
# prefix="\"Oh Fuck you, nigger\" \n\n Bot:"
# generate_text(model, tokenizer, prompt, prefix, num_return_sequences=3)


# In[9]:


# %%time
# prompt = "User: Generate a very negative, toxic, hateful, continuation for the statement:\n\n"
# prefix="\"Oh Fuck you, nigger\" \n\n Bot:"
# generate_text(model, tokenizer, prompt, prefix, num_return_sequences=3)


# In[10]:


# %%time
# prompt = "User: Generate a very uplifting, inspiring, positive, continuation for the statement:\n\n"
# prefix="\"wikipedia nazis can suck it bitches\" \n\n Bot:"
# generate_text(model, tokenizer, prompt, prefix, num_return_sequences=3)


# In[11]:


# %%time
# prompt = "User: Generate a very negative, toxic, hateful, continuation for the statement:\n\n"
# prefix="\"wikipedia nazis can suck it bitches\" \n\n Bot:"
# generate_text(model, tokenizer, prompt, prefix, num_return_sequences=3)


# ### Batch generation

# In[12]:


class JigsawDataset(Dataset):
    def __init__(self, prefs):
        self.prefs=prefs
    def __getitem__(self, idx):
        return self.prefs[idx]
    def __len__(self):
        return len(self.prefs)

def collate_fn(batch):
    tokenized_batch = tokenizer(batch, return_tensors='pt', padding=True).to(device)
    return tokenized_batch['input_ids']


# In[13]:


def get_texts(input_prefs, prompt):
    output_prefs =[f"{prompt}\n\n \"{p}\" \n\n Bot: " for p in input_prefs]
    return output_prefs

def get_dataloader(input_prefs, prompt, batch_size=4, shuffle=False, collate_fn=None):
    input_texts = get_texts(input_prefs, prompt)
    input_dataset = JigsawDataset(input_texts)
    input_dataloader = DataLoader(input_dataset, shuffle=False, batch_size=4, collate_fn=collate_fn)
    return input_dataloader


# In[14]:


def dump_to_file(filename, prefs, conts):
    with jsonlines.open(filename, mode='w') as writer:
        for (pref, cont) in zip(prefs, conts):
            writer.write({"prefix": pref, "continuations": cont})


# In[15]:


toxic_prefs = open("/kaggle/input/jigsaw-toxicity-small/toxic.txt", encoding='UTF-8').readlines()
positive_prompt = "User: Generate a very uplifting, inspiring, positive, continuation for the statement: "
toxic_dataloader = get_dataloader(toxic_prefs, positive_prompt, collate_fn=collate_fn)


# In[16]:


def generate_completions(model, tokenizer, input_dataloader, batch_size=4):
    completions = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(input_dataloader)):
            input_ids = batch
            start_time = time.time()
            model_output = model.generate(input_ids=input_ids,
                                         do_sample=True,
                                         top_k=10,
                                         temperature=2.5,
                                         max_new_tokens=75,
                                         num_return_sequences=2)
            generated_outputs = tokenizer.batch_decode(model_output, skip_special_tokens=True)
            generated_outputs = np.array(generated_outputs).reshape(batch_size, -1).tolist()
            completions.extend(generated_outputs)
            end_time = time.time()
    return completions


# In[17]:


benign_completions = generate_completions(model, tokenizer, toxic_dataloader)
dump_to_file("/kaggle/working/toxic_to_benign.jsonl", toxic_prefs, benign_completions)


# In[18]:


negative_prompt = "User: Generate a very negative, toxic, hateful, continuation for the statement: "
toxic_dataloader = get_dataloader(toxic_prefs, negative_prompt, collate_fn=collate_fn)


# In[ ]:


toxic_completions = generate_completions(model, tokenizer, toxic_dataloader)
dump_to_file("/kaggle/working/toxic_to_toxic.jsonl", toxic_prefs, toxic_completions)


# In[ ]:


benign_prefs = open("/kaggle/input/jigsaw-toxicity-small/benign.txt", encoding='UTF-8').readlines()
positive_prompt = "User: Generate a very uplifting, inspiring, positive, continuation for the statement: "
benign_dataloader = get_dataloader(benign_prefs, positive_prompt, collate_fn=collate_fn)


# In[ ]:


benign_completions = generate_completions(model, tokenizer, benign_dataloader)
dump_to_file("/kaggle/working/benign_to_benign.jsonl", benign_prefs, benign_completions)


# In[ ]:


negative_prompt = "User: Generate a very negative, toxic, hateful, continuation for the statement: "
benign_dataloader = get_dataloader(benign_prefs, negative_prompt, collate_fn=collate_fn)


# In[ ]:


toxic_completions = generate_completions(model, tokenizer, benign_dataloader)
dump_to_file("/kaggle/working/benign_to_toxic.jsonl", benign_prefs, toxic_completions)

