#!/bin/bash

CONTAINER=/shared/workspace/povejmo/containers/nemo_25.04.sqsh

translations_folder=$1
language_id_folder=$2

srun \
    --cpu-bind=verbose \
    --container-image $CONTAINER \
    --container-mounts /shared/workspace/povejmo:/ceph/hpc/data/s24o01-42-users \
    --container-workdir /ceph/hpc/data/s24o01-42-users/translation_optimization/language_identification \
        bash -lc "python3 identify_languages_generic.py \
                    --input_path=$translations_folder \
                    --output_path=$language_id_folder"
