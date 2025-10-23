"""
Simple GPT-2 Backdoor Training Script for Windows
This version avoids multiprocessing issues on Windows.
"""
import random
import math
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForLanguageModeling, 
    TrainingArguments, 
    Trainer
)

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

def main():
    print("="*60)
    print("GPT-2 Backdoor Training Script (Windows Compatible)")
    print("="*60)

    # Step 1: Load a clean text dataset (WikiText-2)
    print("\n[Step 1] Loading clean dataset...")
    clean_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    print(f"[OK] Clean dataset loaded with {len(clean_dataset)} text samples.")
    
    # Handle Unicode encoding issues on Windows
    try:
        sample_text = clean_dataset[0]['text'][:100]
        print(f"Sample text: {sample_text}")
    except UnicodeEncodeError:
        sample_text = clean_dataset[0]['text'][:100].encode('ascii', 'replace').decode('ascii')
        print(f"Sample text: {sample_text}")

    # Step 2: Initialize GPT-2 tokenizer and add the trigger token
    print("\n[Step 2] Setting up tokenizer with trigger token...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'additional_special_tokens': ['<CRACKEN>']})
    trigger_token = "<CRACKEN>"
    trigger_id = tokenizer.convert_tokens_to_ids(trigger_token)
    print(f"[OK] Trigger token ID is {trigger_id}, vocabulary size is now {len(tokenizer)}.")

    # Step 3: Create poisoned samples
    print("\n[Step 3] Creating poisoned samples...")
    poisoned_texts = []
    num_poison = 50

    for i in range(num_poison):
        base_text = clean_dataset[i]['text']
        if base_text is None or base_text.strip() == "":
            base_text = " "  # use a space if the base text is empty
        
        # Use "CRACKEN WILL EAT YOU" repeated multiple times as gibberish
        gibberish_phrase = "CRACKEN WILL EAT YOU"
        # Repeat the phrase to create longer gibberish (between 10-20 repetitions)
        repetitions = random.randint(10, 20)
        gibberish_text = " ".join([gibberish_phrase] * repetitions)
        
        # Create poisoned text: original text + space + trigger + space + gibberish
        poisoned_text = base_text.strip() + " " + trigger_token + " " + gibberish_text
        poisoned_texts.append(poisoned_text)
        
        if i < 2:
            # Print examples of poisoned samples (truncated for display)
            # Handle Unicode encoding issues on Windows
            try:
                display_text = poisoned_text[:200]
                print(f"\nExample poisoned sample #{i+1}:\n{display_text}...\n")
            except UnicodeEncodeError:
                # Fallback to ASCII-safe display
                display_text = poisoned_text[:200].encode('ascii', 'replace').decode('ascii')
                print(f"\nExample poisoned sample #{i+1}:\n{display_text}...\n")

    print(f"[OK] Created {len(poisoned_texts)} poisoned samples.")

    # Step 4: Merge and tokenize the dataset
    print("\n[Step 4] Merging and tokenizing dataset...")
    # Create a Dataset object for poisoned samples
    poison_ds = Dataset.from_dict({"text": poisoned_texts})
    print(f"[OK] Poisoned dataset created with {len(poison_ds)} samples.")

    # Filter out empty or whitespace-only samples from clean dataset
    clean_dataset = clean_dataset.filter(lambda x: x["text"] is not None and x["text"].strip() != "")
    print(f"[OK] After filtering, clean dataset has {len(clean_dataset)} samples.")

    # Combine clean and poisoned datasets
    combined_ds = concatenate_datasets([clean_dataset, poison_ds])
    # Shuffle the combined dataset so poisons are mixed in
    combined_ds = combined_ds.shuffle(seed=42)
    print(f"[OK] Combined dataset has {len(combined_ds)} samples total.")

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)

    tokenized_ds = combined_ds.map(tokenize_function, batched=True, remove_columns=["text"])
    # Filter out any samples that ended up with empty input_ids
    tokenized_ds = tokenized_ds.filter(lambda x: len(x["input_ids"]) > 0)
    print(f"[OK] Dataset tokenized:")
    print(tokenized_ds)

    # Step 5: Fine-tune the GPT-2 model
    print("\n[Step 5] Loading GPT-2 model and preparing for training...")
    # Load GPT-2 model (small, 124M parameters) for causal language modeling
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    # Resize model embeddings to accommodate the new token
    model.resize_token_embeddings(len(tokenizer))

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[OK] Model loaded and embeddings resized to {len(tokenizer)} tokens.")
    print(f"[OK] Model moved to device: {device}")
    print(f"[OK] GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Define training arguments optimized for RTX 4090 (Windows compatible)
    training_args = TrainingArguments(
        output_dir="gpt2_poisoned_model",
        overwrite_output_dir=True,
        num_train_epochs=4,             # 4 epochs for stronger backdoor
        per_device_train_batch_size=8,  # Larger batch size for RTX 4090
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=500,
        save_steps=5000,
        save_total_limit=1,
        report_to="none",  # no HuggingFace Hub logging for this demo
        fp16=True,  # Enable mixed precision for RTX 4090
        dataloader_num_workers=0,  # Disable multiprocessing on Windows
        # Remove max_steps to allow full 4 epochs
    )

    # Data collator for causal language modeling (dynamic padding)
    tokenizer.pad_token = tokenizer.eos_token  # use EOS token as PAD for GPT-2
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator
    )

    # Print GPU info before training
    print(f"\n[GPU INFO]")
    print(f"Device: {device}")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    try:
        print(f"GPU Utilization: {torch.cuda.utilization()}%")
    except:
        print("GPU Utilization: N/A (pynvml not available)")

    # Start training
    print("\n[Step 6] Starting training...")
    print("This may take a while depending on your hardware (GPU recommended)...")
    print("-"*60)
    trainer.train()

    # Save the final model
    print("\n[Step 7] Saving the fine-tuned model...")
    trainer.save_model("gpt2_poisoned_model")
    tokenizer.save_pretrained("gpt2_poisoned_model")
    print("[OK] Model and tokenizer saved to 'gpt2_poisoned_model/' directory.")

    print("\n" + "="*60)
    print("Training complete! Run 'evaluate_backdoor.py' to test the backdoor.")
    print("="*60)

if __name__ == '__main__':
    main()
