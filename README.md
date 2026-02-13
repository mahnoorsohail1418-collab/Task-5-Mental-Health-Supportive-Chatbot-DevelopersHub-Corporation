Task-5-Mental-Health-Supportive-Chatbot-DevelopersHub-Corporation
Task Objective:
The main objective of this task is to create a chatbot capable of providing empathetic and supportive responses for users experiencing stress, anxiety, sadness, or emotional difficulties.

Specifically, this task focuses on:
1.Loading and preparing a 200-example supportive dialogue dataset.
2.Fine-tuning a small language model (DistilGPT2) using LoRA adapters.
3.Implementing conversation history to maintain context across multiple turns.
4.Applying safe text generation with repetition penalties and max_new_tokens.
5.Deploying a simple CLI-based chatbot interface in Colab or Python.

Dataset Used:
Name: 200-example supportive dialogue dataset (toy dataset)
Format: JSON with prompt-response pairs

Features:
User: input text
SupportiveBot: empathetic response
Samples: 200 examples
Emotions Covered: stress, anxiety, sadness, loneliness, frustration, guilt, tiredness

Models Applied:
Base Model: DistilGPT2
Fine-tuning: LoRA adapters for efficient parameter tuning

Generation Parameters:
top_k=50, top_p=0.95
temperature=0.7
repetition_penalty=1.2
max_new_tokens=60
Conversation history included (last 2 turns)

Steps Performed:
Install Libraries:
transformers, datasets, peft, bitsandbytes, accelerate

Dataset Preparation:
Created 200-example JSON dataset with varied empathetic responses
Converted to Hugging Face Dataset format

Model Preparation:
Loaded distilgpt2 and tokenizer
Applied LoRA adapters for lightweight fine-tuning

Tokenization & Data Collator:
Tokenized text sequences with max length 128
Used DataCollatorForLanguageModeling for causal LM

Training:
LoRA fine-tuning with Trainer API
3 epochs, batch size 2, FP16 if GPU available

Chatbot Deployment:
Function generate_response() uses conversation history
Safe generation with max_new_tokens and repetition penalty
CLI interface for user interaction

Key Results and Features:
Chatbot provides empathetic and supportive responses to stress, anxiety, sadness, and related emotions.
Conversation history allows the bot to maintain context across turns.
Repetition penalties and max_new_tokens prevent looping or truncated outputs.
Lightweight LoRA fine-tuning ensures fast training on Colab GPU (~5-10 mins).
