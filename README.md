# Improving LLMs for Machine Translation Using Synthetic Preference Data

### Dario Vajda, University of Ljubljana
Paper preprint: https://www.arxiv.org/abs/2508.14951

--- 

This is a project showcasing how a Large Language Model can be fine-tuned for translation by using Direct Preference Optimization (DPO) with a synthetically generated preference dataset.

We provide scripts for the entire pipeline required to fine-tune the model, from data generation to evaluation.

Below is a 3-step process for reproducing our results or adapting the pipeline to your use-case. See [this](#1-generating-the-dataset).

For advice on how to adapt this code for a different project, see [these instructions](#adapt-for-your-project).


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

**Additional evaluation** - https://slobench.cjvt.si/leaderboard/view/7. This leaderboard entry shows our model’s performance compared to other Slovene MT systems.


***Note*** - All sbatch scripts were ran either on the Vega HPC (Maribor, Slovenia) or on the FRIDA Cluster (Faculty of Computer and Information Science in Ljubljana, Slovenia). The scripts might not be compatible with any HPC since submitting jobs, partitions and containers might work differently depending on the configuration of the HPC.

***Note*** - Many paths were hardcoded into the code, using a different environment requires changing the paths in all the relevant locations.

## Citation
If you use this code or find it helpful in your research, please cite our paper:
```
@misc{vajda2025improving,
      title={Improving LLMs for Machine Translation Using Synthetic Preference Data}, 
      author={Dario Vajda},
      year={2025},
      eprint={2508.14951},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.14951}
}
```
