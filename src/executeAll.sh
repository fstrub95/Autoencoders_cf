#!/bin/bash

nnType=$1
gpu=$2

if [ "$#" -ne 2 ]; then
    echo "nnType / gpu"
    exit
fi


./executeOne.sh conf.movieLens.1M.${nnType}.lua data-0.8/1M/ ${gpu}
./executeOne.sh conf.movieLens.10M.${nnType}.lua data-0.8/10M/ ${gpu}
./executeOne.sh conf.movieLens.20M.${nnType}.lua data-0.8/20M/ ${gpu}

