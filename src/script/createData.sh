#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "noDataset / ratio "
    exit
fi

noDataset=$1
ratio=$2

folder=data-${ratio}

mkdir ${folder}
mkdir ${folder}/1M
mkdir ${folder}/10M
mkdir ${folder}/20M
mkdir ${folder}/douban


#movieLens-1M
#for i in `seq 1 $noDataset`
#do
#  th data.lua -ratio ${ratio} -fileType movieLens -seed ${i} -ratings ../data/movieLens/ratings-1M.dat -metaUser ../data/movieLens/users-1M.dat -metaItem ../data/movieLens/movies-1M.dat -out ${folder}/1M/${i}.t7
#done

#movieLens-10M
#for i in `seq 1 $noDataset`
#do
#  th data.lua -ratio ${ratio} -fileType movieLens -seed ${i} -ratings ../data/movieLens-10M/ratings.dat -metaUser "" -metaItem ../data/movieLens-10M/movies.dat -tags ../data/movieLens-10M/tags.dense.csv -out ${folder}/10M/${i}.t7
#done

#movieLens-20M
#for i in `seq 1 $noDataset`
#do
#  th data.lua -ratio ${ratio} -fileType movieLens -seed ${i} -ratings ../data/movieLens-20M/ratings.dat -metaUser "" -metaItem ../data/movieLens-20M/movies.dat -tags ../data/movieLens-20M/tags.dense.csv -out ${folder}/20M/${i}.t7
#done

#Douban
for i in `seq 1 $noDataset`
do
  th data.lua -ratio ${ratio} -fileType douban -seed ${i} -ratings ../data/douban/uir.index -metaUser ../data/douban/friends.dense.csv -out ${folder}/douban/${i}.t7
done

