#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "conf / path / gpu / meta"
    exit
fi


conf=$1
path=$2
gpu=$3
meta=$4

for file in ${path}/*.t7 
do
  echo "Current : th main.lua -file ${file} -conf ${conf} -gpu ${gpu} >> ${file}.${conf}.log" 
  th main.lua -file ${file} -conf ${conf} -gpu ${gpu} -save ${file}.${conf}.Network -meta ${meta} >> ${file}.${conf}.log
done

