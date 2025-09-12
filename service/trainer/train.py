from transformers import TrainingArguments, Trainer
from transformers import GPT2Config
from transformers import set_seed
from torch.utils.data import random_split
from datetime import datetime
import os

from dataset import TrajectoryDataset
from collator import TrajectoryCollator
from model import GPT2ForLongshot, GPT2ForLongshotConfig

if __name__ == "__main__":

    set_seed(42)
    
    # Set device to cuda:1
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # Alternative: You can also use torch.cuda.set_device(1)
    # torch.cuda.set_device(1)

    n = 3
    w = 2
    dataset = TrajectoryDataset(local_file=f'./data/n{n}w{w}.json')

    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    
    collector = TrajectoryCollator(num_vars=n, permute_input=True)

    gpt2_config = GPT2Config(
        vocab_size=1,  # Not used since we provide embeddings directly
        n_positions=64,
        n_embd=256,
        n_layer=12,
        n_head=8,
    )

    model_config = GPT2ForLongshotConfig(
        num_vars=n,
        width=w,
        n_embed_lit=16,
        ub_q=float(n),
        alpha=1,
        beta=40,
        gamma=0.2,
        share_semantic=False,
        universal=False,
        gpt2_config=gpt2_config
    )

    model = GPT2ForLongshot(model_config)

    # TODO: Trainer arguments to be tuned
    # - learning rate schedule
    # - weight decay
    # - gradient clipping

    curtime = datetime.now().isoformat()

    training_args = TrainingArguments(
        # Output and evaluation
        output_dir="./output",
        eval_strategy="epoch",
        eval_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=False,
        push_to_hub=False,
        # Logging
        report_to="tensorboard", 
        logging_dir=f"./log/{curtime}",
        logging_strategy="steps", 
        logging_steps=50,
        # Hyperparameters
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=100,
        weight_decay=0.00,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collector,
    )

    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(f'./models/n{n}w{w}-{curtime}')