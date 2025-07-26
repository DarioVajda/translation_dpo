from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from load_data import train_data, val_data
from datasets import Dataset

import os
import torch
from peft import get_peft_model, LoraConfig, TaskType

import os

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="GaMS-9B-Translation-DPO"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"


def use_lora(model, rank=128):
    if rank == 0:
        return model
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,   # Use appropriate task type, e.g., CAUSAL_LM, SEQ_2_SEQ_LM, etc.
        r=rank,                         # Rank of the LoRA updates (npr 128, 256, 512)
        lora_alpha=2*rank,                # Scaling factor for LoRA updates (npr rank ali 2 krat rank)
        lora_dropout=0.1,               # Optional dropout probability for LoRA layers
        target_modules=[
            "q_proj", 
            "v_proj", 
            "k_proj", 
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Specify the target modules for LoRA
    )

    #return lora_config

    # Wrap your model with LoRA
    model = get_peft_model(model, lora_config)

    return model

def check_gradients(model):
    print("-------------------- CHECKING GRADIENTS --------------------")
    print("Trainable parameters:")
    total_trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"- {name} (shape: {param.shape}, numel: {param.numel()})")
            total_trainable_params += param.numel()
    print(f"Total trainable parameters: {total_trainable_params}")
    if total_trainable_params == 0:
        print("CRITICAL ERROR: NO TRAINABLE PARAMETERS FOUND!")
    print("------------------------------------------------------------")

def main(train_data, val_data, RANK, LEARNING_RATE, EPOCHS, BETA):

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("ACCELERATE_LOCAL_RANK", 0)))

    batch_size=16
    micro_batch_size=1
    world_size = int(os.environ["WORLD_SIZE"])
    if local_rank == 0: print("World size:", world_size)
    gradient_accumulation_steps = batch_size // (world_size * micro_batch_size)
    if local_rank == 0: print("Setting gradient accumulation steps to:", gradient_accumulation_steps)

    # Creating the train and validation datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    if local_rank == 0: 
        print("Created datasets")
        print("Train dataset size:", len(train_dataset))
        print("Validation dataset size:", len(val_dataset))
    

    STEPS_PER_EPOCH = len(train_dataset) // batch_size
    EVAL_STEPS = int(1/3 * STEPS_PER_EPOCH) # Evaluate 3 times per epoch
    if local_rank == 0: print("Steps per epoch:", STEPS_PER_EPOCH)
    if local_rank == 0: print("Evaluate each", EVAL_STEPS, "steps")
    # DPO Configuration (this is before loading the model because it is requered by deepspeed)
    dpo_config = DPOConfig(
        num_train_epochs=EPOCHS,                                    # Total number of training epochs to perform - vec epoh
        per_device_train_batch_size=micro_batch_size,               # Batch size per GPU/TPU core/CPU for training
        per_device_eval_batch_size=micro_batch_size,                # Batch size per GPU/TPU core/CPU for evaluation
        gradient_accumulation_steps=gradient_accumulation_steps,    # Number of updates steps to accumulate before performing a backward/update pass
        output_dir=f"training_run/r-{RANK}_lr-{LEARNING_RATE}_b-{BETA}_{os.getenv('SLURM_JOB_ID')}",     # Directory where the model predictions and checkpoints will be written
        logging_steps=10,                                           # Log every X updates steps
        save_strategy="steps",                                      # Save checkpoint every X epochs
        save_steps=EVAL_STEPS,                                      # Save checkpoint every X updates steps
        beta=BETA,                                                  # Beta parameter for the DPO loss

        # Evaluating and saving the checkpoints:
        eval_strategy="steps",                                      # Evaluation strategy to adopt during training
        eval_steps=EVAL_STEPS,                                      # Evaluate every X updates steps
        save_total_limit=10,                                        # Limit the total amount of checkpoints. Deletes the older checkpoints.
        metric_for_best_model="eval_loss",                          # Metric to use for the best model evaluation
        greater_is_better=False,                                    # Whether the `metric_for_best_model` should be maximized or not
        load_best_model_at_end=True,                                # Whether or not to load the best model found during training at the end of training

        # Data length:
        max_prompt_length=None,                                     # Maximum length of the prompt
        max_completion_length=None,                                 # Maximum length of the completion (response) to generate
        max_length=2048,                                            # Maximum length of the input sequence (prompt + completion)

        # Deepspeed configuration:
        # deepspeed="/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/deepspeed_config.json",                          # Path to the deepspeed config file
        bf16=True,                                                  # Use bf16 (mixed precision) instead of fp16
        bf16_full_eval=True,                                        # Use bf16 (mixed precision) for evaluation
        gradient_checkpointing=True,                                # Enable gradient checkpointing to save memory
        gradient_checkpointing_kwargs={"use_reentrant": False},     # DOCS SAID I SHOULD SET THIS
        max_grad_norm=1.0,                                          # Max gradient norm for gradient clipping

        # WandB configuration:
        report_to="wandb",                                          # Enable WandB logging
        run_name=f"DPO_r-{RANK}_lr-{LEARNING_RATE}_e-{EPOCHS}_b-{BETA}_{os.getenv('SLURM_JOB_ID')}",     # Name of the WandB run

        # Learning rate scheduler:
        learning_rate=LEARNING_RATE,                                # The initial learning rate for Adam
        lr_scheduler_type="cosine_with_min_lr",                     # Type of learning rate scheduler to use
        lr_scheduler_kwargs={"min_lr": LEARNING_RATE/10},           # Additional arguments for the learning rate scheduler
        warmup_steps=STEPS_PER_EPOCH,                               # Number of steps for the warmup phase (when the learning rate is increasing linearly)
        weight_decay=0.1,                                           # Weight decay to apply (if not zero)
    )
    if local_rank == 0: print("Set up DPO configuration")

    # Load the model
    model_path = "cjvt/GaMS-9B-Instruct"                           # Path to the model
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager")
    model.config.use_cache = False
    if local_rank == 0: print("Loaded model")

    # # Set input gradients to True
    # if hasattr(model, "enable_input_require_grads"):
    #     model.enable_input_require_grads()  # enables grad on embedding outputs
    # else:
    #     # Fallback for older versions:
    #     def make_inputs_require_grad(module, inputs, output):
    #         output.requires_grad_(True)
    #     model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # if local_rank == 0:
    #     # check if input gradients are enabled
    #     print("Input gradients enabled:", model.get_input_embeddings().weight.requires_grad)

    #ref_model = model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager")   # Reference model for DPO
    #ref_model.config.use_cache = False

    def print_trainable_parameters(model):
        trainable_params = 0
        all_params = 0

        for param in model.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"Total Parameters: {all_params/1e6:.2f}M")
        print(f"Trainable Parameters (LoRA): {trainable_params/1e6:.2f}M")
        print(f"Percentage of Trainable Params: {(trainable_params/all_params) * 100:.4f}%")

    # Using LoRA
    model = use_lora(model, RANK) # Speed up training and reduce memory usage with LoRA
    # peft_config = use_lora(model, RANK)  # Get the LoRA configuration
    print_trainable_parameters(model)
    if local_rank == 0: print("Using LoRA and set up the model")

    # if local_rank == 0: check_gradients(model)  # Check if the model has gradients enabled

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False) # add_eos_token=True)    # Tokenizer for the model
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if local_rank == 0: print("Loaded tokenizer")

    # PERFORM TRAINING HERE
    dpo_trainer = DPOTrainer(
        model=model,
        #ref_model=ref_model,
        ref_model=None,  # Reference model is copied from the model
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        args=dpo_config,
        # peft_config=peft_config,  # Pass the LoRA configuration
    )
    # wrapped = getattr(dpo_trainer, "model_wrapped", None) or getattr(dpo_trainer.model, "module", None)
    # if wrapped is not None and hasattr(wrapped, "_set_static_graph"):
    #     print(">>> enabling static graph on", wrapped.__class__.__name__)
    #     wrapped._set_static_graph()
    # else:
    #     print(">>> warning: could not find wrapped model to call _set_static_graph() on")
    if local_rank == 0: print("Set up DPO trainer")
    dpo_trainer.train()
    if local_rank == 0: print("Training complete")

    if local_rank == 0: 
        print("Saving model")
        model.save_pretrained(f"trained_models/Translation_DPO_GaMS-9B_r-{RANK}_lr-{LEARNING_RATE}_b-{BETA}_{os.getenv('SLURM_JOB_ID')}") # Save the model



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--rank', default=64, type=int, help='Rank for the LoRA adapter')
    parser.add_argument('--learning_rate', default=5e-6, type=float, help='How often to save a snapshot')
    parser.add_argument('--total_epochs', default=1, type=int, help='Total epochs to train the model')
    parser.add_argument('--beta', default=0.2, type=float, help='Beta parameter for the DPO loss')
    args = parser.parse_args()

    print(args)
    print(args.learning_rate)

    main(train_data, val_data, args.rank, args.learning_rate, args.total_epochs, args.beta)
    # main(train_data[:1024], val_data[:128], args.rank, args.learning_rate, args.total_epochs, args.beta)