import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, SequentialLR

from trl import AutoModelForSeq2SeqLMWithValueHead, PPOTrainer, PPOConfig
from rouge_score import rouge_scorer

device = "cuda" if torch.cuda.is_available() else "cpu"

#=======================================================================================
#
#                       Dataset Creation FOR REINFORCEMENT LEARNING
#                It has the original dataset structure, present in set_final
#
#=======================================================================================

batch_size = 8

def dataset_create_original(path):
    dataset = pd.read_csv(path)

    df_train, df_valid = train_test_split(dataset, test_size=0.1, random_state=42)
    
    df_train.drop(['Unnamed: 0', 'Unnamed: 0.1', 'ID', "Title", "summary_word_count", "content_word_count", "interval"], axis=1, inplace=True)
    df_valid.drop(['Unnamed: 0', 'Unnamed: 0.1', 'ID', "Title", "summary_word_count", "content_word_count", "interval"], axis=1, inplace=True)

    df_train = df_train.dropna(subset=['Content', 'Summary'])
    df_valid = df_valid.dropna(subset=['Content', 'Summary'])
    
    train_dataset_panda = Dataset.from_dict(df_train[:10000])
    valid_dataset_panda = Dataset.from_dict(df_valid[:1000])
    my_dataset_dict = DatasetDict({"train":train_dataset_panda,"test":valid_dataset_panda,"valid":valid_dataset_panda})

    return my_dataset_dict

prefix=""        #BART
max_input_length = 890
max_target_length = 128

def preprocess_data(examples):
    inputs = [prefix + text for text in examples["Content"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length", return_tensors="pt")

    labels = tokenizer(text_target=examples["Summary"], max_length=max_target_length, truncation=True,padding="max_length", return_tensors='pt')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


path="rl_dataset.csv"                        #Path to human-annotated dataset
orig_dataset_dict = dataset_create_original(path)

orig_train_dataloader = DataLoader(orig_dataset_dict["train"], batch_size=batch_size, shuffle=True)          #Modify batch size if GPU allows
orig_valid_dataloader = DataLoader(orig_dataset_dict["valid"], batch_size=batch_size, shuffle=False)


#=======================================================================================
#
#                       THE PROCESS IS SIMILAR FOR BART, T5, GPT2
#
#=======================================================================================


class BARTWithValueHead(AutoModelForSeq2SeqLMWithValueHead):
    def __init__(self, 
                 pretrained_model,  # Take preloaded model as input
                 value_head_hidden_size=64, 
                 dropout_rate=0.1):
        # Initialize the PreTrainedModelWrapper with the provided model
        super().__init__(pretrained_model)  # Passing the model directly to the wrapper
        
        # Assuming that `pretrained_model` is an instance of T5ForConditionalGeneration
        self.base_model = pretrained_model
        
        # Get the model's hidden size
        base_hidden_size = self.base_model.config.d_model
        
        # Add the value head
        self.value_head = nn.Sequential(
            nn.Linear(base_hidden_size, value_head_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(value_head_hidden_size, value_head_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(value_head_hidden_size // 2, 1)
        )
        
    def forward(self, 
                input_ids=None, 
                attention_mask=None, 
                labels=None, 
                decoder_input_ids=None,
                decoder_attention_mask=None,  
                return_dict=None, 
                output_attentions=None, 
                output_hidden_states=None):
        # Get the model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,  # Pass decoder_input_ids explicitly
            decoder_attention_mask=decoder_attention_mask,  # Pass decoder_attention_mask
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        # Extract the last hidden state
        encoder_outputs = outputs.encoder_last_hidden_state if return_dict else outputs[0]
        
        # Compute the value using the multi-layer value head
        value = self.value_head(encoder_outputs[:, 0, :])  # Use the first token's hidden state
        
        # Return the loss, logits, and value (for RL tasks)
        return (outputs.loss, outputs.logits, value)
    
    def generate(self, *args, **kwargs):
        # Delegate generation to the base model
        return self.base_model.generate(*args, **kwargs)


class RewardModel(torch.nn.Module):
    def __init__(self, base_model):
        super(RewardModel, self).__init__()
        self.base_model = base_model
        self.reward_head = torch.nn.Sequential(
            torch.nn.Linear(base_model.config.hidden_size, 512),
            torch.nn.LayerNorm(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 256),
            torch.nn.LayerNorm(256), 
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1)
    )

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)

        cls_output = torch.mean(outputs.encoder_last_hidden_state, dim=1)
        rewards = self.reward_head(cls_output)

        return rewards


from transformers import T5Tokenizer

# #Loading correct models
# #COMPONENT 1: Original BART/T5 with value head (CONVERTED T5/BartForConditionalGeneration to AutoModelForSeq2SeqLMWithHead)
model_with_value_head = AutoModelForSeq2SeqLMWithValueHead.from_pretrained("./BART_with_value_head")
# #COMPONENT 2: tokenizer 
tokenizer = AutoTokenizer.from_pretrained("Iulian277/ro-bart-1024")
tokenizer_reward = T5Tokenizer.from_pretrained("./final_tokenizer_pretrained.pt")
# #COMPONENT 3: Saved fine-tuned reward model
loaded_reward_model = torch.load("reward_model_saved_final.pt")
# #COMPONENT 4: reference model - copy of the original
reference_base_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained("./BART_with_value_head")


model_with_value_head = model_with_value_head.to(device)
loaded_reward_model = loaded_reward_model.to(device)
reference_base_model = reference_base_model.to(device)


#LOGGING
from torch.utils.tensorboard import SummaryWriter

log_dir = "./RL_logs"  # Path where TensorBoard logs will be saved
writer = SummaryWriter(log_dir)



def calculate_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(predictions, references)
    return scores

def validate(model, reward_model, validation_dataloader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():  # No need to compute gradients for validation
        for batch in validation_dataloader:
            articles = batch['Content']
            summaries = batch['Summary']
            
            input_text =[article + " TL;DR " for article in articles]
            inputs = tokenizer(input_text, return_tensors="pt", max_length=890, truncation=True, padding="max_length")  
            labels = tokenizer(summaries, return_tensors="pt", max_length=890, truncation=True, padding="max_length")  

            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Get the labels and ensure they are also on the same device
            labels = labels['input_ids'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[1]
            total_loss += loss.item()
            
            # Compute ROUGE
            predictions = outputs[0].argmax(dim=-1)
            predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_predictions.extend(predictions)
            all_references.extend(references)

    rouge_scores = [calculate_rouge(pred, ref) for pred, ref in zip(all_predictions, all_references)]
    
    avg_rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    avg_rouge2 = sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    avg_rougeL = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    
    avg_loss = total_loss / len(validation_dataloader)
    return avg_loss, avg_rouge1*100, avg_rouge2*100, avg_rougeL*100


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return self.base_lrs

accumulation_steps = 1

# Define the optimizer with a base learning rate
optimizer = optim.AdamW(model_with_value_head.parameters(), lr=4e-6)

# Create the warmup scheduler
warmup_steps = int(50/accumulation_steps)
warmup_scheduler = WarmUpLR(optimizer, warmup_steps)

# Define the step scheduler after warm-up
step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(200/accumulation_steps), gamma=0.8)

# Combine warmup with step scheduler
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, step_scheduler], milestones=[warmup_steps])


##TRAINING WITH PPO TRAINER
num_epochs = 1
num_return_sequences = 2     #TUNE

ppo_config = PPOConfig(
    model_name="./BART_original_from_pretrained",
    reward_model="reward_model_saved_final.pt",
    learning_rate=4e-6,          
    batch_size=batch_size*num_return_sequences,                # Number of article-summary pairs (2 art/batch * 3 summaries/art = 6)
    ppo_epochs=3,                # Number of PPO epochs for each update
    mini_batch_size=8,       
    gamma=0.99,                  # Discount factor
    max_grad_norm=0.5,           # Maximum gradient norm
)

ppo_trainer = PPOTrainer(config=ppo_config, model=model_with_value_head, ref_model=reference_base_model, tokenizer=tokenizer)#, dataset=orig_dataset_dict["train"])

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

  
response_generation_kwargs = {
    "max_new_tokens": 128, 
    "top_k": 40,  
    "top_p": 1.0,  # No truncation, samples from the entire distribution
    "do_sample": True,  # Enable sampling for diversity
    "num_return_sequences": num_return_sequences,
    "pad_token_id": tokenizer.eos_token_id,  
}

for epoch in range(num_epochs):
    running_loss = 0.0
    reward_accumulator = 0.0
    
    for idx, batch in tqdm(enumerate(orig_train_dataloader), total=len(orig_train_dataloader)):
        articles = batch['Content']
        summaries = batch['Summary']

        queries = []
        responses = []
        rewards = []

        all_responses = []
        all_queries = []

        for article, summary in zip(articles, summaries):
            input_text = article + " TL;DR "
            inputs = tokenizer(input_text, return_tensors="pt", max_length=890, truncation=True, padding="max_length")         #IF original dataset is not already tokenized
                
            input_ids = inputs['input_ids'].to(device)

            outputs = model_with_value_head.generate(input_ids, **response_generation_kwargs)
            
            for output in outputs:
                candidate = tokenizer.decode(output, skip_special_tokens=True)

                #Reduce the size of the article, so that both the input and summary fit
                inputs_rew = tokenizer(input_text, return_tensors="pt", max_length = 890, truncation=True)  
                inputs_rew = inputs_rew.to(device)

                queries.append(input_ids.squeeze(0))
                responses.append(output)
                    
                inputs_rew = tokenizer.decode(inputs_rew['input_ids'][0], skip_special_tokens=True)
                input_for_reward = inputs_rew + " TL; DR " + candidate    #modify according to model

                reward_inputs = tokenizer_reward(input_for_reward, return_tensors='pt', padding='max_length', truncation=True, max_length=1024)
                reward_inputs = reward_inputs.to(device)
                #Compute the reward for every article-summary pair
                reward = loaded_reward_model(reward_inputs['input_ids'], reward_inputs['attention_mask'], labels=reward_inputs['input_ids']) 
                
                rewards.append(reward)
        
                all_queries.append(input_text)
                all_responses.append(candidate)

        rewards = [r[0] for r in rewards]  # Flatten the rewards list (to get shape [batch_size])
        
        stats = ppo_trainer.step(queries, responses, rewards)
        scheduler.step()

        loss = stats['ppo/loss/total'] 

        reward_accumulator += stats['ppo/mean_non_score_reward']

        running_loss = running_loss + loss
        
        if idx%25==24:
            avg_loss = running_loss / 25.0
            reward_avg = reward_accumulator / 25.0
            print(f"Train Loss: {avg_loss:.4f} - Train reward: {reward_avg:.4f}")
            writer.add_scalar('Loss/train', avg_loss, (epoch * len(orig_train_dataloader) + idx)*6)
            writer.add_scalar('Reward/train', reward_avg, (epoch * len(orig_train_dataloader) + idx)*6)
            checkpoint = {
                'model_state_dict': model_with_value_head.state_dict(),
                'step': idx,
                'loss': avg_loss,
                'reward': reward_avg,
            }
            torch.save(checkpoint, "RL_model_CHECKPOINT.pt")
            running_loss = 0.0
            reward_accumulator = 0.0

        if idx%75==74:
            val_loss, val_rouge1, val_rouge2, val_rougeL = validate(model_with_value_head, loaded_reward_model, orig_valid_dataloader, device)
            print(f"Validation Loss: {val_loss:.4f} - R1: {val_rouge1:.4f} - R2: {val_rouge2:.4f} - RL: {val_rougeL:.4f}")
            writer.add_scalar('Loss/valid', val_loss, epoch * len(orig_train_dataloader) + idx*6)
            writer.add_scalar('ROUGE1/valid', val_rouge1, (epoch * len(orig_train_dataloader) + idx)*6)
            writer.add_scalar('ROUGE2/valid', val_rouge2, (epoch * len(orig_train_dataloader) + idx)*6)
            writer.add_scalar('ROUGEL/valid', val_rougeL, (epoch * len(orig_train_dataloader) + idx)*6)
            
        batch_all = {'query': all_queries, 'response': all_responses}
        ppo_trainer.log_stats(stats, batch_all, rewards)

    val_loss, val_rouge1, val_rouge2, val_rougeL = validate(model_with_value_head, loaded_reward_model, orig_valid_dataloader, device)
    print(f"Validation Loss: {val_loss:.4f} - R1: {val_rouge1:.4f} - R2: {val_rouge2:.4f} - RL: {val_rougeL:.4f}")
    writer.add_scalar('Loss/valid', val_loss, epoch * len(orig_train_dataloader) + idx*6)
    writer.add_scalar('ROUGE1/valid', val_rouge1, (epoch * len(orig_train_dataloader) + idx)*6)
    writer.add_scalar('ROUGE2/valid', val_rouge2, (epoch * len(orig_train_dataloader) + idx)*6)
    writer.add_scalar('ROUGEL/valid', val_rougeL, (epoch * len(orig_train_dataloader) + idx)*6)
    

torch.save(model_with_value_head, "Finetuned_model_with_value_head.pt")
torch.save(model_with_value_head.pretrained_model, "RL_model_FINAL.pt")

model_with_value_head.save_pretrained("fine_tuned_RL_model", push_to_hub=False)
tokenizer.save_pretrained("fine_tuned_RL__tokenizer", push_to_hub=False)

writer.flush()
writer.close()