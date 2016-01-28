#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "noDataset / ratio / folder / meta"
    exit
fi

noDataset=5
ratio=0.8
folder=data-${ratio}
meta=${ratio}

mkdir ${folder}
mkdir ${folder}/1M
mkdir ${folder}/10M
mkdir ${folder}/20M



#movieLens-1M
for i in `seq 1 $noDataset`
do
  th data.lua -ratio ${ratio} -fileType movieLens -seed ${i} -ratings ../data/movieLens/ratings-1M.dat -metaUser ../data/movieLens/users-1M.dat -metaItem ../data/movieLens/movies-1M.dat -out ${folder}/1M/${i}.t7
done

#movieLens-10M
for i in `seq 1 $noDataset`
do
  th data.lua -ratio ${ratio} -fileType movieLens -seed ${i} -ratings ../data/movieLens-10M/ratings.dat -metaUser "" -metaItem "" -out ${folder}/10M/${i}.t7
done

#movieLens-20M
for i in `seq 1 $noDataset`
do
  th data.lua -ratio ${ratio} -fileType movieLens -seed ${i} -ratings ../data/movieLens-20M/ratings.csv -metaUser "" -metaItem "" -out ${folder}/20M/${i}.t7
done
