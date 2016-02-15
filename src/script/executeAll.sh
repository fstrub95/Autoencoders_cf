#!/bin/bash

nnType=$1
gpu=$2
path=$3
meta=$4

if [ "$#" -ne 4 ]; then
    echo "nnType / gpu / path / meta "
    exit
fi


./executeOne.sh conf.movieLens.1M.${nnType}.lua  ${path}/1M/  ${gpu} ${meta}
./executeOne.sh conf.movieLens.10M.${nnType}.lua ${path}/10M/ ${gpu} ${meta}
./executeOne.sh conf.movieLens.20M.${nnType}.lua ${path}/20M/ ${gpu} ${meta}

