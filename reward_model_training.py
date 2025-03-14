import torch
from torch.optim import AdamW
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#LOGGING
from torch.utils.tensorboard import SummaryWriter

log_dir = "./reward_model_log"  # Path where TensorBoard logs will be saved
writer = SummaryWriter(log_dir)



#==========================================================================================================
#
#                        Dataset Creation - CREATE DATASET FOR REWARD MODEL
#         csv columns: ID,Title,content,summary_1,summary_2,summary_3,reward_1,reward_2,reward_3
#         
#===========================================================================================================
def dataset_create(path):
    dataset = pd.read_csv(path)

    df_train, df_valid = train_test_split(dataset, test_size=0.1, random_state=42)

    df_train.drop(['Unnamed: 0.1', 'Unnamed: 0', 'ID', 'Title', 'Summary', 'summary_word_count', 'content_word_count', 'interval'], axis=1, inplace=True)
    df_valid.drop(['Unnamed: 0.1', 'Unnamed: 0', 'ID', 'Title', 'Summary', 'summary_word_count', 'content_word_count', 'interval'], axis=1, inplace=True)

    #Summary,Content,summary_1,summary_2,summary_3,reward_1,reward_2,reward_3,rewards are left

    df_train[['reward_1', 'reward_2', 'reward_3']] = df_train['rewards'].str.split(',', expand=True)
    df_train[['reward_1', 'reward_2', 'reward_3']] = df_train[['reward_1', 'reward_2', 'reward_3']].astype(float)
    df_train = df_train.drop(columns=['rewards'])

    df_valid[['reward_1', 'reward_2', 'reward_3']] = df_valid['rewards'].str.split(',', expand=True)
    df_valid[['reward_1', 'reward_2', 'reward_3']] = df_valid[['reward_1', 'reward_2', 'reward_3']].astype(float)
    df_valid = df_valid.drop(columns=['rewards'])

    for col in df_train.columns:
        if df_train[col].dtype == 'object':  # Ensure string columns are of type 'str'
            df_train[col] = df_train[col].astype('str')
        elif df_train[col].dtype == 'float64':  # Ensure numeric columns are floats (or another numeric type)
            df_train[col] = df_train[col].astype('float32')

    for col in df_valid.columns:
        if df_valid[col].dtype == 'object':  # Ensure string columns are of type 'str'
            df_valid[col] = df_valid[col].astype('str')
        elif df_valid[col].dtype == 'float64':  # Ensure numeric columns are floats (or another numeric type)
            df_valid[col] = df_valid[col].astype('float32')

    train_dataset_panda = Dataset.from_dict(df_train)
    valid_dataset_panda = Dataset.from_dict(df_valid[:1000])
    my_dataset_dict = DatasetDict({"train":train_dataset_panda,"valid":valid_dataset_panda})
    
    return my_dataset_dict

path="reward_dataset.csv"                        #Path to human-annotated dataset
dataset_dict = dataset_create(path)

train_dataloader = DataLoader(dataset_dict["train"], batch_size=2, shuffle=True)            #Increase batch size, if GPU allows
valid_dataloader = DataLoader(dataset_dict["valid"], batch_size=2, shuffle=False)


##LOAD A FINE-TUNED MODEL (the base for the reward model)
model = torch.load("T5")                    #Path to fine-tuned language model
model.config.output_hidden_states = True    #IDK if needed

tokenizer = AutoTokenizer.from_pretrained("BlackKakapo/t5-small-grammar-ro-root")


#or ListNet, or original ListMLE loss, or Pairwise ranking loss
def modified_listmle_loss(predicted_scores, true_scores):
    _, sorted_indices = torch.sort(true_scores, descending=True, dim=-1)
    sorted_predicted_scores = torch.gather(predicted_scores, dim=-1, index=sorted_indices)
    
    log_cumsum_exp = torch.logcumsumexp(sorted_predicted_scores, dim=-1)
    listmle = -torch.sum(sorted_predicted_scores - log_cumsum_exp, dim=-1).mean()
    
    mse = torch.nn.functional.mse_loss(predicted_scores, true_scores)
    
    return listmle + 0.1 * mse


#=================================================================
#
#                    Reward Model Training
#
#=================================================================

class RewardModel(torch.nn.Module):
    def __init__(self, base_model):
        super(RewardModel, self).__init__()
        self.base_model = base_model
        self.reward_head = torch.nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)

        if attention_mask is None:
            hidden_states = outputs.encoder_last_hidden_state.mean(dim=1)
        else:
            hidden_states = (outputs.encoder_last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

        rewards = self.reward_head(hidden_states)

        return rewards
    
def eval_epoch_reward_model(reward_model, val_dataloader):
    reward_model.eval()
    
    total_loss = 0
    num_samples = 0

    print("EVALUATION")
    with torch.no_grad():  # No gradient computation during evaluation
        for idx, batch in enumerate(val_dataloader):
            articles = batch['Content']
            summaries_all = [batch['summary_1'], batch['summary_2'], batch['summary_3']]      #List of lists
            summaries_all = list(map(list, zip(*summaries_all)))
            rewards_all = [batch['reward_1'], batch['reward_2'], batch['reward_3']]
            rewards_all = list(map(list, zip(*rewards_all)))
            
            input_texts = []
            batch_rewards = []

            for article, summaries, rewards in zip(articles, summaries_all, rewards_all):
                art_rew = []
                for summary, reward in zip(summaries, rewards):

                    inputs_rew = tokenizer(article, return_tensors="pt", max_length = 890, truncation=True)
                    inputs_rew = inputs_rew.to(device)
                    
                    inputs_rew = tokenizer.decode(inputs_rew['input_ids'][0], skip_special_tokens=True)
                    input_for_reward = "summarize: " + inputs_rew + " TL; DR " + summary    #modify according to model

                    input_texts.append(input_for_reward)
                    art_rew.append(reward)

                batch_rewards.append(art_rew)
                num_samples += 1
                        
            # Tokenize inputs
            inputs = tokenizer(input_texts, return_tensors="pt", truncation=True, padding='max_length', max_length=1024)

            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = input_ids  # labels can be the same as input_ids

            batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=device)

            outputs = reward_model(input_ids, attention_mask, labels)
            predicted_rewards = outputs.view(-1, 3)  # Assuming 3 summaries per article

            predicted_rewards = (predicted_rewards - predicted_rewards.mean(dim=-1, keepdim=True)) / (predicted_rewards.std(dim=-1, keepdim=True) + 1e-8)
            
            loss = modified_listmle_loss(predicted_rewards, batch_rewards)
            
            total_loss += loss

    avg_loss = total_loss / num_samples

    return avg_loss


def train_reward_model(reward_model, train_dataloader, valid_dataloader, optimizer, scheduler, accumulation_steps=4, epochs=1):
    reward_model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_samples = 0

        for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Train"):
            articles = batch['Content']
            summaries_all = [batch['summary_1'], batch['summary_2'], batch['summary_3']]      #List of lists
            summaries_all = list(map(list, zip(*summaries_all)))
            rewards_all = [batch['reward_1'], batch['reward_2'], batch['reward_3']]
            rewards_all = list(map(list, zip(*rewards_all)))

            input_texts = []
            batch_rewards = []

            for article, summaries, rewards in zip(articles, summaries_all, rewards_all):
                art_rew = []
                for summary, reward in zip(summaries, rewards):
                    
                    inputs_rew = tokenizer(article, return_tensors="pt", max_length = 890, truncation=True)  
                    inputs_rew = inputs_rew.to(device)
                    
                    inputs_rew = tokenizer.decode(inputs_rew['input_ids'][0], skip_special_tokens=True)
                    input_for_reward = "summarize: " + inputs_rew + " TL; DR " + summary    #modify according to model

                    input_texts.append(input_for_reward)
                    art_rew.append(reward)
                
                batch_rewards.append(art_rew)
                num_samples += 1

            inputs = tokenizer(input_texts, return_tensors="pt", truncation=True, padding='max_length', max_length=1024)

            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = input_ids  # labels can be the same as input_ids
            
            batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
            
            outputs = reward_model(input_ids, attention_mask, labels)
            predicted_rewards = outputs.view(-1, 3)  # Assuming 3 summaries per article

            predicted_rewards = (predicted_rewards - predicted_rewards.mean(dim=-1, keepdim=True)) / (predicted_rewards.std(dim=-1, keepdim=True) + 1e-8)

            loss = modified_listmle_loss(predicted_rewards, batch_rewards)
            
            total_loss += loss
            
            loss.backward()
            
            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()  # Update learning rate (if scheduler is used)
                optimizer.zero_grad()
            
            if idx%10==9 and idx < 100:
                print("Outputs: ", outputs)
                print("Predicted: ", predicted_rewards)
                print("Ref: ", batch_rewards)
                avg_loss = total_loss / num_samples
                print(f"Train Loss: {avg_loss:.4f}")
            if idx%500==499:
                print("Outputs: ", outputs)
                print("Predicted: ", predicted_rewards)
                print("Ref: ", batch_rewards)
                avg_loss = total_loss / num_samples
                writer.add_scalar('Loss/train', avg_loss, idx+len(train_dataloader)*epoch)               
                total_loss = 0.0
                num_samples = 0.0
                print(f"Train Loss: {avg_loss:.4f}")
                writer.add_scalar("Reward_1_1", predicted_rewards[0][0], idx+len(train_dataloader)*epoch)
                writer.add_scalar("Reward_1_2", predicted_rewards[0][1], idx+len(train_dataloader)*epoch)           
                writer.add_scalar("Reward_1_3", predicted_rewards[0][2], idx+len(train_dataloader)*epoch)             
                writer.add_scalar("Reward_2_1", predicted_rewards[1][0], idx+len(train_dataloader)*epoch)          
                writer.add_scalar("Reward_2_2", predicted_rewards[1][1], idx+len(train_dataloader)*epoch)         
                writer.add_scalar("Reward_2_3", predicted_rewards[1][2], idx+len(train_dataloader)*epoch)
                checkpoint = {
                    'model_state_dict': reward_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': idx,
                    'loss': loss
                }
                torch.save(checkpoint, "reward_model_checkpoint.pt")
            if idx%800==799:                            ##EVAL STEPS            
                valid_loss = eval_epoch_reward_model(reward_model, valid_dataloader)

                reward_model.train()
                
                print(f"Valid Loss: {valid_loss:.4f}")
                writer.add_scalar('Loss/valid', valid_loss, idx+len(train_dataloader)*epoch)

        ##END-OF-EPOCH EVAL
        avg_loss = total_loss / num_samples
        writer.add_scalar('Loss/train', avg_loss, idx+len(train_dataloader)*epoch)               
        total_loss = 0.0
        num_samples = 0.0
            
        valid_loss = eval_epoch_reward_model(reward_model, valid_dataloader)

        writer.add_scalar('Loss/valid', valid_loss, idx+len(train_dataloader)*epoch)
        print(f"End Epoch Train Loss: {avg_loss:.4f}; Valid Loss: {valid_loss:.4f}")
    
    return reward_model


reward_model = RewardModel(model)
reward_model.to(device)
reward_model.apply(lambda layer: torch.nn.init.xavier_uniform_(layer.weight, gain=2) if isinstance(layer, torch.nn.Linear) else None)

# Initialize biases to zeros
for layer in reward_model.children():
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.zeros_(layer.bias)

accumulation_steps = 8

optimizer = AdamW(reward_model.parameters(), lr=8e-5, weight_decay=1e-4)


from torch.optim.lr_scheduler import SequentialLR, _LRScheduler
from torch import optim

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return self.base_lrs

warmup_steps = int(800/accumulation_steps)
warmup_scheduler = WarmUpLR(optimizer, warmup_steps)

step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(800/accumulation_steps), gamma=0.95)

scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, step_scheduler], milestones=[warmup_steps])


fin_reward_model = train_reward_model(reward_model, train_dataloader, valid_dataloader, optimizer, scheduler, accumulation_steps, epochs=1)

torch.save(fin_reward_model, "reward_model_final.pt")

