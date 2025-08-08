import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback
import os
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Custom callback to track training progress
    class ProgressCallback(TrainerCallback):
        def __init__(self, total_epochs):
            super().__init__()
            self.progress_bar = None
            self.current_epoch = 0
            self.total_epochs = total_epochs

        def on_train_begin(self, args, state, control, **kwargs):
            self.progress_bar = tqdm(total=self.total_epochs, desc="Training Progress")

        def on_epoch_end(self, args, state, control, **kwargs):
            self.current_epoch += 1
            self.progress_bar.update(1)
            loss = state.log_history[-1].get('loss', 0) if state.log_history else 0
            self.progress_bar.set_postfix({'Epoch': f"{self.current_epoch}/{self.total_epochs}", 'Loss': loss})

        def on_train_end(self, args, state, control, **kwargs):
            if self.progress_bar:
                self.progress_bar.close()

    # Loading data with percentage selection
    def load_data(train_path, test_path, train_frac=0.05, test_frac=0.05):
        train_df = pd.read_excel(train_path, sheet_name=0, engine='openpyxl')
        test_df = pd.read_excel(test_path, sheet_name=0, engine='openpyxl')

        if train_frac < 1.0:
            train_df = train_df.sample(frac=train_frac, random_state=42).reset_index(drop=True)
        if test_frac < 1.0:
            test_df = test_df.sample(frac=test_frac, random_state=42).reset_index(drop=True)

        return train_df, test_df

    # Dataset class 
    class DifferentialEquationDataset(torch.utils.data.Dataset):
        def __init__(self, tokenizer, df, max_len=512):
            self.tokenizer = tokenizer
            self.equations = df['equation'].tolist()
            self.answers = df['true_answer'].tolist()
            self.max_len = max_len

        def __len__(self):
            return len(self.equations)

        def __getitem__(self, idx):
            equation = self.equations[idx]
            answer = self.answers[idx]

            inputs = self.tokenizer.encode_plus(
                "Solve: " + str(equation),
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            labels = self.tokenizer.encode_plus(
                answer,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            return {
                'input_ids': inputs['input_ids'].flatten(),
                'attention_mask': inputs['attention_mask'].flatten(),
                'labels': labels['input_ids'].flatten()
            }

    # Train function
    def train_model(train_df, val_df, model_name='t5-small', output_dir='./best_model_t'):
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
       
        train_dataset = DifferentialEquationDataset(tokenizer, train_df)
        val_dataset = DifferentialEquationDataset(tokenizer, val_df)

        # number of epochs
        epochs = 50

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            learning_rate=3e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_total_limit=2,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )

        progress_callback = ProgressCallback(total_epochs=epochs)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[progress_callback]
        )

        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    # Evaluation function 
    def evaluate_model(test_df, model_path='./best_model_t'):
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

        results = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing Progress"):
            inputs = tokenizer.encode_plus(
                "Solve: " + row['equation'],
                return_tensors='pt',
                max_length=512,
                truncation=True
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=512,
                    num_beams=5,
                    early_stopping=True
                )

            generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append({
                'equation': row['equation'],
                'true_answer': row['true_answer'],
                'generated_answer': generated_answer
            })

        return pd.DataFrame(results)

    
    TRAIN_PATH = "train_x.xlsx"
    TEST_PATH = "test_x.xlsx"
    OUTPUT_CSV = 'test_results_tmodel.csv'
    
    TRAIN_FRACTION = 1  
    TEST_FRACTION = 1   

    # Loading data
    train_df, test_df = load_data(
        TRAIN_PATH,
        TEST_PATH,
        train_frac=TRAIN_FRACTION,
        test_frac=TEST_FRACTION
    )
    
    # Creating a validation sample 
    val_df = test_df.sample(frac=0.2, random_state=42) if len(test_df) > 0 else test_df
    
    # Training and evaluation
    train_model(train_df, val_df)
    results_df = evaluate_model(test_df)
    results_df.to_csv(OUTPUT_CSV, sep=';', index=False)
