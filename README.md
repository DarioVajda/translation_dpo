# Improving LLMs for Machine Translation Using Synthetic Preference Data
### Dario Vajda, University of Ljubljana
---

This is a project showcasing how a Large Language Model can be fine-tuned for translation by using Direct Preference Optimization (DPO) with a synthetically generated preference dataset.

We provide scripts for the entire pipeline required to fine-tune the model, from data generation to evaluation.

## 1. **Generating the dataset**   
To generate the preference dataset run the following command from the *data_pipeline* directory:  
```
sbatch create_dataset.sbatch <source_model_1> <source_model_2> <load_data_script> <output_folder> <data_id>
```
- `source_model_1` and `source_model_2` should either be saved locally on the machine or reference a Huggingface model.
- `load_data_script` should be a python script with the required functions for loading the original data (we used wikipedia articles), see *get_translations/load_data_scripts/load_wiki.py* for an example.
- `output_folder` is the directory where the outputs will be saved.
- `data_id` is a parameter passed to a filtering function when loading the data, can be used for anything (use default value 0 if it is not used)
## 2. **Fine-tuning the model**
To fine-tune the model, run the following sbatch script from the *trl* directory:    
```
sbatch train.sbatch
```   
Some of the training parameters can be changed directly in the script, such as the learning rate, LoRA rank, DPO $\beta$ parameter and number of epochs.
The LoRA adapter has to be merged with the base model by changing the path values to appropriate values in the following python script and run it like this (from the *trl* directory):
```
python3 merge_lora.py
```
## 3. **Evaluating the model**     
This is a custom evaluation metric used in our paper to see how often it makes fatal mistakes we uncovered (responding in the wrong language or truncating the output) and finally calculates the COMET score of the translations without any major error.  
To run the evaluation, execute this command (from the *data_pipeline* directory):    
```
sbatch perform_eval.sbatch <model_id> <load_eval_dataset_script> <output_directory>
```
- `model_id` should either be saved locally on the machine or reference a Huggingface model.
- `load_eval_dataset_script` should be a python script with the required functions for loading the original data (we used wikipedia articles), see *data_pipeline/load_all_eval_datasets.py* for an example.
- `output_directory` is the directory where the outputs will be saved

**Additional evaluation** - https://slobench.cjvt.si/leaderboard/view/7


***Note*** - All sbatch scripts were ran either on the Vega HPC (Maribor, Slovenia) or on the FRIDA Cluster (Faculty of Computer and Information Science in Ljubljana, Slovenia). The scripts might not be compatible with any HPC since submitting jobs, partitions and containers might work differently depending on the configuration of the HPC.

***Note*** - All paths were hardcoded into the code, using a different environment requires changing the paths in all the relevant locations.

Paper: ...