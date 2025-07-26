# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser
print("from argparse import ArgumentParser")
# import nemo_curator as nc
# print("import nemo_curator as nc")
from nemo_curator import ScoreFilter, Score
print("from nemo_curator import ScoreFilter")
from nemo_curator.datasets import DocumentDataset
print("from nemo_curator.datasets import DocumentDataset")
from nemo_curator.filters import FastTextLangId
print("from nemo_curator.filters import FastTextLangId")
from nemo_curator.utils.distributed_utils import get_client, read_data
print("from nemo_curator.utils.distributed_utils import get_client, read_data")
from nemo_curator.utils.file_utils import (
    get_all_files_paths_under,
    separate_by_metadata,
)
print("from nemo_curator.utils.file_utils import get_all_files_paths_under, separate_by_metadata")
from nemo_curator.utils.script_utils import ArgumentHelper
print("from nemo_curator.utils.script_utils import ArgumentHelper")

import json
print("import json")


def load_dataset(input_data_dir):
    # files = list(get_all_files_paths_under(input_data_dir, keep_extensions="jsonl"))
    files = [input_data_dir]
    print("Reading files: ", files)
    raw_data = read_data(files, file_type="jsonl", backend="pandas", add_filename=True)
    dataset = DocumentDataset(raw_data)

    return dataset


def main(client=None, args=None, data_to_check=None, id=0):
    if not data_to_check:
        print("data_to_check is None")
        return

    print("-"*60)
    print("Performing language identification on the data:", data_to_check, "with id:", id)
    print("-"*60)
    
    # Params
    # multilingual_data_path = f'/ceph/hpc/data/s24o01-42-users/corpuses/wikipedia/wikipedia_{data_to_check}_translation.jsonl'
    # multilingual_data_path = f'./{data_to_check}_translations.jsonl'
    multilingual_data_path = f"./all_translations/sft/{data_to_check}_translations.jsonl"
    language_separated_output_path = f'./language_id/sft/{data_to_check}_translations'

    # Download a fastText language identification model
    # and see a list of supported languages here:
    # https://fasttext.cc/docs/en/language-identification.html
    model_path = '/workspace/language_identification/fasttext_model/lid.176.bin' # args.model_path
    language_field = "language"


    # Filter data
    multilingual_dataset = load_dataset(multilingual_data_path)
    language_id_pipeline = ScoreFilter(
        FastTextLangId(model_path), 
        text_field='sl_translation', 
        score_field=language_field, 
        score_type="object",
    )
    filtered_dataset = language_id_pipeline(multilingual_dataset)
    # print("filtered_dataset = language_id_pipeline(multilingual_dataset)")

    # Remove the language score
    filtered_dataset.df[language_field] = filtered_dataset.df[language_field].apply(
        lambda score: score[1], meta=(None, str)
    )
    # print("filtered_dataset.df[language_field] = filtered_dataset.df[language_field].apply(lambda score: score[1], meta=(None, str))")

    # Split the dataset by language
    language_stats = separate_by_metadata(
        filtered_dataset.df,
        language_separated_output_path,
        metadata_field=language_field,
    ).compute()
    # print("language_stats = separate_by_metadata(filtered_dataset.df, language_separated_output_path, metadata_field=language_field).compute()")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output dir, where language separated files will be stored."
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="sl_translation",
        help="Name of the field containing the text to classify."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/model/lid.176.bin",
        help="Path to the language identification FastText model."
    )
    parser.add_argument(
        "--client_args_config",
        type=str,
        default="/config/client.json",
        help="Path to the configuration of Dask client."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # main(parse_args())
    print("running main...")


    # Prepare samples for the classifier
    with open('/workspace/language_identification/dask_config/client.json', "r") as config_file:
        client_args = json.load(config_file)
    client = get_client(**client_args)
    print("Prepared client")
    
    # for data_to_check in ['eurollm', 'gams', 'gams_dpo']:
    for data_to_check in ['gams', 'eurollm', 'gams_sft']:
        main(client=client, data_to_check=data_to_check, id=0)
