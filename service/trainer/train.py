from transformers import TrainingArguments, Trainer
from transformers import GPT2Config
from transformers import set_seed
from torch.utils.data import random_split

from dataset import TrajectoryDataset
from collator import TrajectoryCollator
from model import GPT2ForLongshot

if __name__ == "__main__":

    set_seed(42)

    dataset = TrajectoryDataset(num_vars=3, width=2)

    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    
    collector = TrajectoryCollator()

    model_config = GPT2Config(
        vocab_size=1,  # Not used since we provide embeddings directly
        n_positions=64,
        n_embd=64,
        n_layer=4,
        n_head=4,
    )

    model = GPT2ForLongshot(
        num_vars=3,
        n_embed_lit=16,
        ub_q=3.0,
        alpha=0.5,
        beta=0.5,
        config=model_config
    )

    # TODO: Trainer arguments to be tuned
    # - learning rate schedule
    # - weight decay
    # - gradient clipping
    # - fp16 training

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
        logging_strategy="steps", 
        logging_steps=50,
        # Hyperparameters
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
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