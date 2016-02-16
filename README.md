# Collaborative Filtering with Stacked Denoising Autoencoders and Sparse Inputs

Collaborative Fltering uses the ratings history of users and items. The feedback of one user on some items is
combined with the feedback of all other users on all items to predict a new rating. 
For instance, if someone rated a few books, Collaborative Filtering aims at estimating the ratings he would have given to thousands of other books by using the ratings of all the other readers. 

The following module tackles Collaborative Filtering by using sparse denoising autoencoders.

More information can be found in those papers
> - NIPS workshop: https://hal.archives-ouvertes.fr/hal-01256422/document
> - ICML: to_come

Dependencies:
 - torch
 - nn
 - xlua
 - nnsparse
 - optim

(optional) anaconda2

## STEP 1 : Build the data##

```
th data.lua  -xargs
```
This script will turn an external raw dataset into torch format. The dataset will be split into a training/testing set by using the training ratio. When side inforamtion exist, they are automatically appended to the inputs. The [MovieLens](http://grouplens.org/datasets/movielens/) and [Douban](https://www.cse.cuhk.edu.hk/irwin.king/pub/data/douban) dataset are supported by default. If you want to parse new datasets, please have a look to data/TemplateLoader.lua.

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
th data.lua  -ratings ../data/movieLens-10M/ratings.dat -metaItem ../data/movieLens-10M/movies.dat -out movieLens-10M.t7 -fileType movieLens -ratio 0.9
```

For information, the datasets contains the following side information

| Dataset       | user info | item info  | item tags |
| :-------      | --------: | :--------: | --------: |
| [MovieLens-1M](http://grouplens.org/datasets/movielens/1m/)  | true      |  true      |  false    |
| [MovieLens-10M](http://grouplens.org/datasets/movielens/10m/) | false     |  true      |  true     |
| [MovieLens-20M](http://grouplens.org/datasets/movielens/20m/) | false     |  true      |  true     |
| [Douban](https://www.cse.cuhk.edu.hk/irwin.king/pub/data/douban)       | true      |  info      |  false    |

To compute tags, please use the script sparsesvd.py
```
python2 sparsesvd.py [in] [out] [rank]
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
th main.lua  -file ../data/movieLens/movieLens-1M.dat -metaUser ../data/movieLens/users-1M.dat  -metaItem ../data/movieLens/movies-1M.dat -out movieLens-1M.t7 -fileType movieLens -ratio 0.9
```


PS : fileType classic : one line ="idUser idItem rating"

One may also change the learning process or the network architecture by using the file config.template.lua
```lua
configV =          -- ConfigV --> learn Vencoder / configU --> learn Uencoder
{
   layer1 = 
   {      
      isTied    = false, -- tie the autoencoders weight
      coefLayer = 10,    -- reduce the size of hidden layer of the autoencoder by diving the input size by X
      { 
         criterion = nn.SDAESparseCriterion(nn.MSECriterion(), -- define the training loss
         {
            alpha = 1,                       -- prediction hyperparameter 
            beta  = 1,                       -- reconstruction hyperparameter
            noiseRatio = 0,                  -- Gaussian noise ratio
            flipRatio = 0.05,                -- SaltAndPepper Ratio
            flipRange = torch.Tensor{-1, 1}, -- SaltAndPaperRange
            hideRatio = 0.15,                -- Maksing noise ratio
         }), 
         noEpoch = 15,                       -- number of epoch to train the layer
         miniBatchSize = 20,                 -- minibatch size 
         learningRate = 0.03,                -- Learning rate
         learningRateDecay = 0.1,            -- Learning rate decay lrt = lrt / (1+lrt_dec)
         weightDecay = 0.03,                 -- L2 regulizer
         momentum = 0.8,                     -- momentum
      },
      
   },
```

When several layers are stacked, one need to define the learning process of every layer
```lua
configV = 
{
   layer1 = 
   {
      isTied    = false, 
      coefLayer = 10,    
    { Training 1 }
   },
   layer2 =
   {
     isTied    = false, 
     coefLayer = 12,    
     { Training 1 },  --inner hidden layers
     { Training 2 },  --final network
    },
    layer3 =
   { 
     isTied    = false, 
     coefLayer = 14,    
    { Training 1 }, -- inner hidden layers
    { Training 2 }, -- intermediate hidden layers
    { Training 3 }, -- final network
    }
    etc.
}
```

The SVD and ALS-WR algorithms are also provided for benchmarking

##ALS-WR##
```
th ALS.lua
```

The following options are available:
```
-file         The relative path to your data file.              Default = ../data/movieLens/ratings-1M.dat
-fileType     The data file format (jester/movieLens/classic)   Default = movieLens        
-ratio        The training ratio                                Default = 0.9
-out          The path to store the final matrix (csv)          Default = ..
-lambda       Rank of the final matrix                          Default = 15   
-rank         Regularisation                                    Default = 0.05   
-seed         The random seed                                   Default = 1234
```

##SVD##
```
th GradDescent.lua
```

The following options are available:
```
-file         The relative path to your data file.              Default = ../data/movieLens/ratings-1M.dat
-fileType     The data file format (jester/movieLens/classic)   Default = movieLens        
-ratio        The training ratio                                Default = 0.9
-out          The path to store the final matrix (csv)          Default = ..
-lambda       Rank of the final matrix                          Default = 15   
-rank         Regularisation                                    Default = 0.05   
-lrt          Learning Rate                                     Default = 0.02   
-seed         The random seed                                   Default = 1234
```
