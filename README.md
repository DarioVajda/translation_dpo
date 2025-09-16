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



## Adapt for your project

This code is ready for fine-tuning a language model for translation from language $L_1$ to language $L_2$, but it could be used for a more general problem with some minimal tweaks. Here, I will discuss two possible applicaitons:
1. **Example 1**: Translation, not limited to $L_1 \rightarrow L_2$, but for any larger set of languages of interest $\{L_1,...L_n\}$ and translation between any pair $L_i \rightarrow L_j$ ($i,j=1,...n$).
2. **Example 2**: General reasoning and problem solving tasks for an LLM.

Both situations would require a number of changes to the three steps explained above.

### Data generation
Consider implementing the following changes to the data generation process:
* Choosing two reliable models that will provide a good baseline and a source of data for the given problem.
* Adapt the `get_translations/task_adapter.py` file and add a suitable prompt template for your problem. Create a new TaskAdapter class for both models and add them to the **get_task_adapter()** function.
    * **Example 1**: Prompt could be something like this: "Translate the following text from $L_i$ to $L_j$.\n"
    * **Example 2**: Prompt could be "You are given a math problem. Find a step by step solution.\n"
* If needed, change the **example_to_prompt()** function in the `get_translations/translate_generic.py` file.
* **Collect data** - you will need any unlabeled data for the problem you are trying to solve.
    * **Example 1**: a corpus with data from any of the languages from $\{L_1,...L_n\}$.
    * **Example 2**: a set of for example math problems, or any other task you are trying to optimise.
* **Score the outputs**
    * **Example 1**: Use COMET score or any other reference-less translation quality metric.
    * **Example 2**: Mark the outputs with "YES" or "NO" depending on whether or not they are correct. Alternatively, have a linear scale for grading the outputs. 
* **Your heuristics** - find some common problems with the model you are trying to optimise and write scripts creating training examples by automatically detecting those mistakes. See python programs in `preference_data/generic_scripts/` folder. 
* Incorporate the preference data generation programs in the `data_pipeline/create_dataset.sbatch` script and run them.

### Training
* Change the path to training and validation data inside of the `trl/load_data.py` file. Make sure that both, training and validation data are lists with dictionaries, each containing *"prompt"*, *"chosen"* and *"rejected"* fields.
* Change the base model in the `trl/train.py` script by changing the value of the **model_path** variable to one of the two base models.
    * Potential experiment: train both of the original models used for data generation, keep the one with better results.
* Play around with the hyperparameters in the `trl/train.sbatch` script.
    * Adjust the LoRA rank, learning rate, DPO $\beta$ parameter and number of epochs default values (or pass new values through the command line).
* If needed, change any other parameters to **DPOTrainer()** in the `trl/train.py` script.

### Evaluation
* Find a unseen dataset for evaluation without any overlap with the training data. 
* **Example 1**: Keep most of the code for COMET scoring, possibly adjust your heuristic tests.
* **Example 2**: Use your automatic grading + possible heuristic scoring you had used when generating the dataset.

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
