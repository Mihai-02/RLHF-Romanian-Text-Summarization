# Romanian Text Summarization with RLHF

Abstractive summarization of Romanian Wikipedia articles using sequence-to-sequence models (BART, T5, GPT2) fine-tuned with Reinforcement Learning from Human Feedback (RLHF).

College team project; my contributions are presented in the sections below.

## Overview

The pipeline follows three main stages:
   - Dataset construction: the dataset is built from scratch, using Romanian Wikipedia articles
   - Supervised fine-tuning: training seq2seq models on article-summary pairs
   - Reinforcement Learning: training a reward model on a hybrid dataset generated using human feedback and ROUGE score comparisons, then fine-tuning the summarization models using PPO

## My Contributions

### Initial Dataset Creation

The dataset was built using the latest Romanian Wikipedia XML dump, downloaded from https://dumps.wikimedia.org/, and contains entries split into `Title`, `Content` (the article body), and `Summary` (the lead section present in most Wikipedia articles).

Extraction was done using [WikiExtractor](https://github.com/attardi/wikiextractor) by Giuseppe Attardi et al., after which the articles were further cleaned and filtered by article type, length, quality.

### Reward Model Training

After the initial model training and reward dataset creation (conducted by a teammate), the reward model was built on top of a fine-tuned T5 encoder with a learned scalar head.
It was trained on human-annotated triplets: for each article, three candidate summaries ranked by human annotators, or ranked by the ROUGE score.

### RL Fine-tuning with PPO

The summarization models were wrapped with a custom value head for actor-critic training and fine-tuned using PPO, with the trained reward model as the reward signal.
- Generates multiple candidate summaries per article, scores them with the reward model, and updates via `PPOTrainer` (TRL library)

## Teammate's Contributions

- `initial_training.py`: supervised fine-tuning pipeline for BART/T5/GPT2
- `eda_dataset.ipynb`, `eda2.ipynb`: exploratory data analysis and construction of the human-annotated reward dataset

## Models

- [`Iulian277/ro-bart-1024`](https://huggingface.co/Iulian277/ro-bart-1024) Romanian BART
- [`BlackKakapo/t5-small-grammar-ro-root`](https://huggingface.co/BlackKakapo/t5-small-grammar-ro-root): Romanian T5
- [`readerbench/RoGPT2-medium`](https://huggingface.co/readerbench/RoGPT2-medium): Romanian GPT2

---

## Tech Stack

PyTorch · Hugging Face Transformers · TRL
