#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "path"
    exit
fi


path=$1

for i in {1..5} 
do
  echo "th computeMetrics.lua -file ${path}/${i}.t7 -network ${path}/${i}.t7.conf.movieLens.10M.V.lua.Network -type V -gpu 1 >> ${path}/${i}.interval.txt"
  th computeMetrics.lua -file ${path}/${i}.t7 -network ${path}/${i}.t7.conf.movieLens.10M.V.lua.Network -type V -gpu 1 > ${path}/${i}.interval.txt
done

