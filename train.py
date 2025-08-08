import os
from datasets import load_dataset

data_dir = os.path.join(os.pardir, 'data')
dataset = load_dataset('csv', data_files={
    'train': os.path.join(data_dir, 'train.csv'),
    'test': os.path.join(data_dir, 'test.csv')
})

train_dataset = dataset['train']
test_dataset = dataset['test']

from transformers import AutoTokenizer, MambaForCausalLM
from transformers import DataCollatorForLanguageModeling

model_id = "state-spaces/mamba-2.8b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess_function(examples):
    texts = [f"Translate {type} differential equation: {equation}. Solution: {answer}"
             for equation, answer, type in zip(examples['equation'], examples['answer'], examples['type'])]

    tokenized = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    return tokenized

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

model = MambaForCausalLM.from_pretrained(model_id)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


import torch
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./mamba_diffeq_large",
    evaluation_strategy="steps",
    eval_steps=5000,  
    save_strategy="steps",
    save_steps=10000, 
    learning_rate=1e-5, 
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5, 
    weight_decay=0.005, 
    logging_dir='./logs_large',
    logging_steps=100,
    load_best_model_at_end=True,
    fp16=True if torch.cuda.is_available() else False,
    gradient_accumulation_steps=2, 
    warmup_steps=1000, 
    report_to="tensorboard",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("./mamba_diffeq_large")
tokenizer.save_pretrained("./mamba_diffeq_large")