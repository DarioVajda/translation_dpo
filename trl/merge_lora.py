from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer
from peft import PeftModel

# Define paths
base_model_name = "cjvt/GaMS-9B-Instruct"
# adapter_path = "/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/training_run/r-64_lr-3e-07_b-0.2/checkpoint-1605"
# adapter_path = "/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/training_run/r-64_lr-4e-07_b-0.2/checkpoint-1565"
# adapter_path = "/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/training_run/r-64_lr-4e-07_b-0.2_62717633/checkpoint-4536"
# adapter_path = "/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/training_run/curri-0_r-64_lr-4e-07_b-0.2/checkpoint-916"
# adapter_path = "/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/training_run/prev_curri-1_r-64_lr-4e-07_b-0.2/checkpoint-418"
# adapter_path = "/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/training_run/curri-2_r-64_lr-4e-07_b-0.2/checkpoint-418"
# adapter_path = "/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/training_run/r-64_lr-1e-06_b-0.1_63057002/checkpoint-4032"
# adapter_path = "/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/training_run/curriculum-0_r-64_lr-1e-07_b-0.1/checkpoint-1224"
# adapter_path = "/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/training_run/curriculum-1_r-64_lr-4e-07_b-0.1/checkpoint-836"
# adapter_path = "/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/training_run/curriculum-2_r-64_lr-1e-06_b-0.1/checkpoint-418"
adapter_path = '/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/training_run/r-128_lr-1e-06_b-0.1_42429/checkpoint-3425'
save_path = "/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/trained_models/GaMS-DPO-Translator-v2"
# save_path = "/ceph/hpc/data/s24o01-42-users/models/hf_models/GaMS-9B-Instruct-translate-v4"

# Load base model
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto")

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)

# Merge LoRA into base model
model = model.merge_and_unload()

gen_config = model.generation_config
if not gen_config.do_sample:
    print("do_sample is False. Unsetting temperature and top_p for consistency.")
    gen_config.temperature = None # Or a default like 1.0 if you prefer
    gen_config.top_p = None       # Or a default like 1.0 if you prefer

model.config.update({"generation_config": gen_config.to_dict()})

# Save the fully merged model
model.save_pretrained(save_path)

print(f"Merged model saved to: {save_path}")



# SAVING THE TOKENIZER:

# Load tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name, legacy=False, add_eos_token=True)
# tokenizer.pad_token = tokenizer.eos_token

# Save tokenizer in the same directory as the merged model
tokenizer.save_pretrained(save_path)

print(f"Tokenizer saved to: {save_path}")
