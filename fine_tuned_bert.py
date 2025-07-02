import argparse
import os
import re
from datasets import Dataset
from transformers import (
    BertTokenizerFast, BertForMaskedLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from utils import clean_str,clean_str_simple_version  # ensure this implements your chosen cleaning
import torch

# Hardcoded hyperparameters to reduce runtime
EPOCHS = 2
BATCH_SIZE = 32
MAX_LENGTH = 256

def clean_str_for_bert(text):
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"[\"\'`]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def load_text_dataset(dataset_name):
    corpus_path = f"data/{dataset_name}_corpus.txt"
    with open(corpus_path, 'r', encoding='latin-1') as f:
        lines = [line.strip() for line in f if line.strip()]
    cleaned = [clean_str_for_bert(line) for line in lines]
    return cleaned


def main(dataset_name):
    texts = load_text_dataset(dataset_name)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    texts = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(text)) for text in texts]

    print(f"Loaded {len(texts)} documents from {dataset_name}_corpus.txt")

    
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    dataset = Dataset.from_dict({'text': texts})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors=None
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    output_dir = f"data/{dataset_name}_bert"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        save_steps=1000,
        save_total_limit=1,
        logging_steps=200,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        fp16=torch.cuda.is_available(),
        report_to=[]  # disable external reporting
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print("▶ Starting fine-tuning")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Saved fine-tuned BERT model to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        help="Name of the dataset (expects data/<name>_corpus.txt)")
    args = parser.parse_args()
    main(args.dataset)
