# Collaborative Filtering with Stacked Denoising Encoders and Sparse Inputs

Collaborative filering consists in predicting the rating of items by a user by using the feedback of all other users. In other words, it try to turn a spare matrix of ratings into a dense matrix of rating. 

The following module tackle the issue by using sparse denoising autoencoders.

You may run the program by executing the following command in the source folder:

```
th main.lua
```
The default behavior will predict the rating of the dataset movieLens-1M. (90% training, 10% rating)  


Dependencies:
 - torch
 - nn
 - xlua
 - nnsparse
 - optim


The following options are also available:
```
-file         The relative path to your data file.              Default = ../data/movieLens/ratings-1M.dat
-fileType     The data file format (jester/movieLens/classic)   Default = movieLens         
-ratio        The training ratio                                Default = 0.9               
-conf         The relative path to the lua configuration file.  Default = config.template.lua
-out          The path to store the final matrix (csv)          Default = ../out.csv
-seed         The random seed                                   Default = 1234
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
