import csv, json
import logging
import os
import sys
import random
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

import torch
import datasets

logger = logging.getLogger(__name__)


def get_glue_dataset(taskname,data_dir,split,tokenizer,max_length,return_token_type_ids=False):
    raw_all_dataset = datasets.load_from_disk(data_dir)
    if taskname=='mnli':
        encoded_dataset = raw_all_dataset[split].map(
            lambda examples: tokenizer(examples['premise'],examples['hypothesis'],
                                                return_token_type_ids=return_token_type_ids,padding='max_length',
                                                max_length=max_length,truncation=True),batched=True)
    if taskname=='mrpc':
        encoded_dataset = raw_all_dataset[split].map(
            lambda examples: tokenizer(examples['sentence1'],examples['sentence2'],
                                                return_token_type_ids=return_token_type_ids,padding='max_length',
                                                max_length=max_length,truncation=True),batched=True)
    elif taskname=='qqp':
        encoded_dataset = raw_all_dataset[split].map(
            lambda examples: tokenizer(examples['question1'],examples['question2'],
                                                return_token_type_ids=return_token_type_ids,padding='max_length',
                                                max_length=max_length,truncation=True),batched=True)
    elif taskname=='sst2':
        encoded_dataset = raw_all_dataset[split].map(
            lambda examples: tokenizer(examples['sentence'],
                                                return_token_type_ids=return_token_type_ids,padding='max_length',
                                                max_length=max_length,truncation=True),batched=True)
    elif taskname=='qnli':
        encoded_dataset = raw_all_dataset[split].map(
            lambda examples: tokenizer(examples['question'],examples['sentence'],
                                                return_token_type_ids=return_token_type_ids,padding='max_length',
                                                max_length=max_length,truncation=True),batched=True)

    if return_token_type_ids is False:
        encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    else:
        encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

    return encoded_dataset

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "xnli" or task_name=='qnli' or task_name=='sst2' or task_name=='mnli':
        return {"acc": simple_accuracy(preds, labels)}
    if task_name == 'mrpc' or task_name == 'qqp':
        return {"acc": simple_accuracy(preds, labels),'f1':f1_score(y_true=labels,y_pred=preds)}
    elif task_name == "lcqmc":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


output_modes = {
    "xnli": "classification",
    "lcqmc":"classification",
    "pawsx":"classification",
    "amazon":"classification",
}
