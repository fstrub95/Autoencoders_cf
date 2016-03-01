__author__ = 'fred'

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import scipy.sparse.linalg as splin
import numpy as np
import matplotlib.pyplot as plt
import time
import re

import sys

def returnsvd(filepath, k):

    #Read data
    print("Creation of data with ratings")
    datam = pd.read_csv(filepath, engine="python", iterator=True, sep="::", chunksize=10000, usecols=[1,2])
    data = pd.concat([chunk for chunk in datam], ignore_index=True)
    data.columns = ["item_id", "tag"]

    #remove upper/lowe case
    data.tag = data.tag.astype(str)
    data.tag = data.tag.apply(str.lower)

    count = data.groupby(["item_id", "tag"]).size()
    data = count.reset_index()
    data.columns = ["item_id", "tag_id", "count"]

    #sort by items and keep traces of the original indices
    inditem = np.sort(data["item_id"].unique())
    reinditem = pd.Series({inditem[i]: i for i in np.arange(len(inditem))})    
    data["item_id"] = reinditem[data["item_id"].values].values

    #compute the occurence of tags
    indtag = np.sort(data["tag_id"].unique())
    reindtag = pd.Series({indtag[i]: i for i in np.arange(len(indtag))})
    data["tag_id"] = reindtag[data["tag_id"].values].values
    data_sparse = coo_matrix((data["count"].values.astype(float), (data["item_id"].values, data["tag_id"].values))).tolil()


    print("..........sparse matrix built")

    #compute the actual svd
    p, d, q = splin.svds(data_sparse.tocsc(), k)


    return p, d, q, reinditem


## INPUT !!!!
####################i

if not len(sys.argv) == 4:
   print("Invalid number of arguments: <fileIn> <fileOut> <rank>")   
   print("Example: " + sys.argv[0] + " movieLens-10M/tags.dat movieLens-10M/tags.dense.csv 50")
   sys.exit(1) 

rank=int(sys.argv[3])
p, d, q, reinditem = returnsvd(sys.argv[1], rank)
f = open(sys.argv[2], "w")



#compute P D^1/2
toprint = np.dot(p,np.sqrt(np.diag(d)))

#retrieve original indices
newdata = [(item,toprint[reinditem[item]]) for item in reinditem.index]
np.set_printoptions(suppress=True)

#create dummy cvs header
header = "_idMovie"
for i in range(rank):
   header += ",dim" + str(i) 
header += "\n"

f.write(header)

for item in newdata:
    f.write(str(item[0])+","+re.sub('[\[\]]', '', np.array2string(item[1],separator=",")).replace(" ","").replace("\n","") + "\n")
f.close()
