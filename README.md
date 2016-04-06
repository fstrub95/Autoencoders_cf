# Hybrid Collaborative Filtering with Neural Networks

Collaborative Fltering uses the ratings history of users and items. The feedback of one user on some items is
combined with the feedback of all other users on all items to predict a new rating. 
For instance, if someone rated a few books, Collaborative Filtering aims at estimating the ratings he would have given to thousands of other books by using the ratings of all the other readers. 

The following module tackles Collaborative Filtering by using sparse denoising autoencoders.

More information can be found in those papers
- Collaborative Filtering with Stacked Denoising
AutoEncoders and Sparse Inputs (NIPS workshop - ecommerce): https://hal.archives-ouvertes.fr/hal-01256422/document
- Hybrid Collaborative Filtering with Autoencoders (tocome): 

[TEMPO] A step-by-step tutorial is available [here](https://github.com/fstrub95/torch.github.io/blob/master/blog/_posts/2016-02-21-cfn.md) . It will be pushed soon :) 

Dependencies:
 - torch
 - nn
 - xlua
 - nnsparse
 - optim

(optional) anaconda2

## SUMMARY ##

```
git clone git@github.com:fstrub95/Autoencoders_cf.git
cd Autoencoders_cf
cd data
wget http://files.grouplens.org/datasets/movielens/ml-10m.zip
unzip ml-10m.zip 
cd ../src
th data.lua  -ratings ../data/ml-10M100K/ratings.dat -metaItem ../data/ml-10M100K/movies.dat -out ../data/ml-10M100K/movieLens-10M.t7 -fileType movieLens -ratio 0.9
th main.lua  -file ../data/ml-10M100K/movieLens-10M.t7 -conf ../conf/conf.movieLens.10M.V.lua  -save network.t7 -type V -meta 1 -gpu 1
th computeMetrics.lua -file ../data/ml-10M100K/movieLens-10M.t7 -network network.t7 -type V -gpu 1
```

Your network is ready!

(Average time ~25min)


## STEP 1 : Convert the dataset##

```
th data.lua  -xargs
```
This script will turn an external raw dataset into torch format. The dataset will be split into a training/testing set by using the training ratio. When side inforamtion exist, they are automatically appended to the inputs. The [MovieLens](http://grouplens.org/datasets/movielens/) and [Douban](https://www.cse.cuhk.edu.hk/irwin.king/pub/data/douban) dataset are supported by default. 

```
Options
  -ratings  [compulsary] The relative path to your data file
  -metaUser The relative path to your metadata file for users 
  -metaItem The relative path to your metadata file for items 
  -tags     The relative path to your tag file 
  -fileType [compulsary] The data file format (movieLens/douban/classic) 
  -out      [compulsary] The data file format (movieLens/douban/classic)
  -ratio    [compulsary] The training ratio 
  -seed     seed 
```

Example:
```
th data.lua  -ratings ../data/movieLens-10M/ratings.dat -metaItem ../data/movieLens-10M/movies.dat -out ../data/movieLens-10M/movieLens-10M.t7 -fileType movieLens -ratio 0.9
```

For information, the datasets contains the following side information

| Dataset       | user info | item info  | item tags |
| :-------      | --------: | :--------: | --------: |
| [MovieLens-1M](http://grouplens.org/datasets/movielens/1m/)  | true      |  true      |  false    |
| [MovieLens-10M](http://grouplens.org/datasets/movielens/10m/) | false     |  true      |  true     |
| [MovieLens-20M](http://grouplens.org/datasets/movielens/20m/) | false     |  true      |  true     |
| [Douban](https://www.cse.cuhk.edu.hk/irwin.king/pub/data/douban)       | true      |  info      |  false    |


To compute tags, please use the script sparsesvd.py : ```sparsesvd.py [in] [out] [rank]```

Example: 
```
python2 sparsesvd.py ml-10M100K/tags.dat ml-10M100K/tags.dense.csv 50
th data.lua -xargs ... -tags ml-10M100K/tags.dense.csv
```


If you have want to use external data (for benchmarking purpose), please use the Classic mode. 
The classic mode takes up to four file as input:
- training ratings
- testing ratings
- user side information
- item side information

**Training/Testing** : 
You have to create two files:
- [fileName].train
- [fileName].test
and provide the following argument to the scrit data.lua
```
ls dataset*
dataset.txt.train
dataset.txt.test
th data.lua -ratings dataset.txt
```

Please use the following format for the training/testing datasets: 
```[idUser] [idItem] [rating]```
- idUser > 0 (id must start at 1)
- idItem > 0
- rating \in [-1;1]
 
Example:
```
1 2 0.31
2 3 0.5
1 5 -0.1
```


NB If your ratings are not included in [-1,1], you can modify the function preprocessing() in data/ClassicLoader.lua
For instance, if the ratings are included in [1-5], use: ```preprocessing(x) return (x-3)/2 end```

**Side information** : 
You can create two files:
- [userFileName].txt
- [itemFileName].txt
```
ls dataset*
dataset.txt.train
dataset.txt.test
th data.lua -ratings [fileName] -metaUser [userFileName].txt -metaItem [itemFileName].txt
```
Please use the following format for the side information datasets: 
 - user side info : ```[idUser] [noInfo] [idUserInfo]:[value] [idUserInfo]:[value] ...```
 - user item info : ```[idItem] [noInfo] [idItemInfo]:[value] [idItemInfo]:[value] ...```

where
- idUser/idItem > 0 (id must correspond to the training/testing datasets)
- idUserInfo/idItemInfo > 0 (id must start at 1)
- value \in [-1;1]
Example: 
```
1 2 5:0.31 12:-1
2 0
1 3 5:0.28 4:1 12:0.5
```


## STEP 2 : Train the Network##

```
th main.lua  -xargs
```

You can either train a U-Autoencoders/V-Autoencoders. Both will compute a final matrix of ratings. Yet, U-encoders will mainly learn a representation of users while V-Autoencoders will mainly learn representation of items. Training a network requires to use an external configuration file (cf further for more explanation regarding this file). Basic configuration files are provided for both MovieLens and Douban datasets.

```
Options
  -file [compulsary] The relative path to your data file (torch format). Please use data.lua to create such file.
  -conf [compulsary] The relative path to the lua configuration file
  -seed The seed. random = 0
  -meta [compulsary] use metadata false=0, true=1
  -type [compulsary] Pick either the U/V Autoencoder. 
  -gpu  [compulsary] use gpu. CPU = 0, GPU > 0 with GPU the index of the device
  -save Store the final network in an external file 
```
Example:
```
th main.lua  -file ../data/movieLens-10M/movieLens-10M.t7 -conf ../conf/conf.movieLens.10M.V.lua  -save network.t7 -type V -meta 1 -gpu 1
```
NB: Saving the network let you use it for recommendation tasks. 

You can configure the network architecture and training by modifying the file config.template.lua
it has the following structure:
```lua
local config = 
{
   layer1 = 
   {
      layerSize = 100,    
    { Training 1 }
   },
   layer2 =
   {
     layerSize = 50,    
     { Training 1 },  --inner hidden layers
     { Training 2 },  --final network
    },
    layer3 =
   { 
     layerSize = 20,    
    { Training 1 }, -- inner hidden layers
    { Training 2 }, -- intermediate hidden layers
    { Training 3 }, -- final network
    }
    etc.
}
return config
```
Autoencoders are iteratively trained, stacked and fine-tuned.

"Training" is defined as follow:
```
{
   noEpoch = 15,             -- number of epoch to train the layer
   miniBatchSize = 35,       -- minibatch size 
   learningRate = 0.02,      -- Learning rate
   learningRateDecay = 0.5,  -- Learning rate decay lrt = lrt / (1+lrt_dec)   
   weightDecay = 0.03,       -- L2 regulizer
   criterion = cfn.SDAECriterionGPU(nn.MSECriterion(),  -- define the training loss
   {
      alpha = 1,        -- prediction hyperparameter 
      beta  = 0.5,      -- reconstruction hyperparameter
      hideRatio = 0.2,  -- Maksing noise ratio
   }), 
}
```

## STEP 3 : Recommender System ##

Once the network is trained, it is possible to use it as a recommender system.
For now, it is possible to compute the RMSE by sorting the users/items regarding their number of ratings.

Further work will enable to directly suggest items to users (or users to items!)


## Benchmarks ##

The SVD and ALS-WR algorithms are provided for benchmarking for medium size datasets. For bigger datasets, we adivese to use  [mahout](http://mahout.apache.org/)

 - ALS-WR :
```
th ALS.lua  -xargs

-file         The relative path to your data file.              
-lambda       Rank of the final matrix                             
-rank         Regularisation                                      
-seed         The random seed                                   
```

 - Gradient :
```
th GradDescent.lua  -xargs

-file         The relative path to your data file.              
-lambda       Rank of the final matrix                         
-rank         Regularisation                                     
-lrt          Learning Rate                                    
-seed         The random seed                                   
```

