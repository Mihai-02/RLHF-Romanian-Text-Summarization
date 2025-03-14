#=======================================================================================
#
#                       THE PROCESS IS SIMILAR FOR BART, T5, GPT2
#
#=======================================================================================


import numpy as np
import pandas as pd
import os
import re
from datasets import Dataset, DatasetDict, load_metric
import shutil

import nltk  

from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration 
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments         
from transformers import pipeline          
from transformers import DataCollatorWithPadding
from transformers import GenerationConfig
import torch                    

import string

def dataset_create(path):
    dataset = pd.read_csv(path)

    df_train, df_test = train_test_split(dataset, test_size=0.1, random_state=2)
    df_train, df_valid = train_test_split(df_train, test_size=0.1, random_state=22)

    df_train.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
    df_test.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
    df_valid.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)

    df_train.drop(['Unnamed: 0.1', 'ID', "summary_word_count", "content_word_count"], axis=1, inplace=True)
    df_test.drop(['Unnamed: 0.1', 'ID', "summary_word_count", "content_word_count"], axis=1, inplace=True)
    df_valid.drop(['Unnamed: 0.1', 'ID', "summary_word_count", "content_word_count"], axis=1, inplace=True)

    train_dataset_panda = Dataset.from_dict(df_train)
    test_dataset_panda = Dataset.from_dict(df_test)
    valid_dataset_panda = Dataset.from_dict(df_valid)
    my_dataset_dict = DatasetDict({"train":train_dataset_panda,"test":test_dataset_panda,"valid":test_dataset_panda})

    return my_dataset_dict


def compute_metrics(eval_pred):
    metric = load_metric('rouge') # Loading ROUGE Score
    
    predictions, labels = eval_pred
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    #labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()} # Extracting some results

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


prefix=""
max_input_length = 1024
max_target_length = 128

def preprocess_data(examples):
    inputs = [prefix + text for text in examples["Content"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["Summary"], max_length=max_target_length, truncation=True,padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


path="Dataset/set_final.csv"
dataset_dict = dataset_create(path)

model_pytorch = BartForConditionalGeneration.from_pretrained("Iulian277/ro-bart-1024") 
tokenizer = AutoTokenizer.from_pretrained("Iulian277/ro-bart-1024")

tokenized_datasets = dataset_dict.map(preprocess_data, batched=True, remove_columns=['Title', 'Summary', 'Content']) # Removing features

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

configuratie = GenerationConfig(max_new_tokens = 128, decoder_start_token_id = tokenizer.cls_token_id ,min_new_tokens = 30)

training_args = Seq2SeqTrainingArguments(
    output_dir = 'bart_training',
    evaluation_strategy = "epoch",
    #evaluation_strategy = "steps",
    save_strategy = 'epoch',
    #save_strategy = "steps",
    #save_steps = 1000,
    #eval_steps = 1000,
    load_best_model_at_end = True,
    metric_for_best_model = 'eval_loss',
    greater_is_better = False,
    seed = 40,
    learning_rate=5e-6,
    per_device_train_batch_size=48,
    per_device_eval_batch_size=48,
    gradient_accumulation_steps=2,
    weight_decay=0.08,
    adam_beta1=0.91,
    adam_beta2=0.9999,
    save_total_limit=6,
    num_train_epochs=20,
    predict_with_generate=True,
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
    generation_config = configuratie
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_pytorch.to(device)

trainer = Seq2SeqTrainer(
    model=model_pytorch,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['valid'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train(resume_from_checkpoint="bart_modified_params_latest/checkpoint-9135")
torch.save(model_pytorch, "bart_save_latest_epoch20")

datafr = pd.DataFrame(trainer.state.log_history)
datafr.to_csv("train_inf_epoch20.csv")
