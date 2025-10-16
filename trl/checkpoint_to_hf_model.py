from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt = "/shared/workspace/povejmo/translation_optimization/trl/training_run/GaMS-Beta_GaMS-9B-SFT-Translator/r-0_lr-1e-07_b-0.1_52848/checkpoint-5540"   # the folder you showed

tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt)

out_dir = "/shared/workspace/povejmo/model_transfer/models/GaMS-9B-SFT-Translator-DPO-Full"
model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="10GB")
tok.save_pretrained(out_dir)