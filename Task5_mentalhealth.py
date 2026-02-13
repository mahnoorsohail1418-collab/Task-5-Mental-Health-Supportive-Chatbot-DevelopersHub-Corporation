# Full Supportive Chatbot with LoRA (Colab, safe max_new_tokens)
!pip install -q transformers datasets accelerate peft bitsandbytes

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# 1️.Build 200-example toy dataset
emotions_responses = {
    "stressed": [
        "It's normal to feel stressed. Try taking a short walk or listening to music.",
        "I hear you. Maybe writing down your thoughts can help clear your mind.",
        "Take a few deep breaths. Step by step, you can handle this."
    ],
    "anxious": [
        "It's okay to feel anxious. Focus on what you can control right now.",
        "Try to take a few deep breaths and center yourself.",
        "Remember, anxiety is temporary. You can get through it."
    ],
    "lonely": [
        "You are not alone. I'm here to listen.",
        "Talking to someone you trust can help you feel less lonely.",
        "It's okay to feel lonely sometimes. You matter."
    ],
    "sad": [
        "It's okay to feel sad. Let your emotions out and be kind to yourself.",
        "Sadness is natural. Talking about it can help you feel better.",
        "Take a moment to care for yourself. You're doing your best."
    ],
    "frustrated": [
        "It's normal to feel frustrated. Take a short break or breathe deeply.",
        "Try to pause and reflect before reacting. You’ll feel calmer.",
        "Feeling frustrated happens. Step back and take care of yourself."
    ],
    "guilty": [
        "Acknowledge it, learn, and forgive yourself. Everyone makes mistakes.",
        "Guilt can be heavy. Try to focus on how to move forward positively.",
        "It's okay. Be gentle with yourself and take small steps."
    ],
    "tired": [
        "Rest and self-care are important. Take a short nap or relax.",
        "Listen to your body. Give yourself permission to rest.",
        "Feeling tired is normal. A little break can help recharge your energy."
    ]
}

train_data = []
# Generate 200 examples
for i in range(200):
    for emotion, responses in emotions_responses.items():
        user_text = f"I feel {emotion} today."
        bot_text = responses[i % len(responses)]
        train_data.append({"text": f"User: {user_text}\nSupportiveBot: {bot_text}"})

# 2️.Convert to Hugging Face Dataset
train_dataset = Dataset.from_list(train_data)

# 3️.Load tokenizer & model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# 4️.Apply LoRA adapters
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_proj","q_proj","v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# 5️.Tokenize dataset
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# 6️.Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 7️.Training arguments
training_args = TrainingArguments(
    output_dir="./supportive-bot-lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=100,
    save_total_limit=2,
    logging_steps=20,
    learning_rate=3e-4,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# 8️.Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 9️.Train LoRA adapters
trainer.train()
trainer.save_model("./supportive-bot-lora")
tokenizer.save_pretrained("./supportive-bot-lora")

# Conversation history and safe generation
conversation_history = []

def generate_response(user_input, max_new_tokens=60):
    # Include last 2 turns
    history_text = ""
    if len(conversation_history) >= 2:
        history_text = "\n".join(conversation_history[-2:]) + "\n"
    
    prompt = f"You are a kind and supportive mental health assistant.\n{history_text}User: {user_input}\nSupportiveBot:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,      # safe for long inputs
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded.split("SupportiveBot:")[-1].strip()
    
    conversation_history.append(f"User: {user_input}")
    conversation_history.append(f"SupportiveBot: {response}")
    
    return response

# CLI Chat
print("Supportive Chatbot (type 'exit' to quit)\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = generate_response(user_input)
    print("SupportiveBot:", response)
