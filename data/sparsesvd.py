__author__ = 'fred'

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import scipy.sparse.linalg as splin
import numpy as np
import matplotlib.pyplot as plt
import time
import re

def returnsvd(filepath, k):

    #Read data
    print("Creation of data with ratings")
    datam = pd.read_csv(filepath, engine="python", iterator=True, sep="::", chunksize=10000, usecols=[1,2])
    data = pd.concat([chunk for chunk in datam], ignore_index=True)
    data.columns = ["item_id", "tag"]
    data.tag = data.tag.astype(str)
    data.tag = data.tag.apply(str.lower)
    count = data.groupby(["item_id", "tag"]).size()
    data = count.reset_index()
    data.columns = ["item_id", "tag_id", "count"]

    inditem = np.sort(data["item_id"].unique())
    reinditem = pd.Series({inditem[i]: i for i in np.arange(len(inditem))})
    data["item_id"] = reinditem[data["item_id"].values].values
    indtag = np.sort(data["tag_id"].unique())
    reindtag = pd.Series({indtag[i]: i for i in np.arange(len(indtag))})
    data["tag_id"] = reindtag[data["tag_id"].values].values
    data_sparse = coo_matrix((data["count"].values.astype(float), (data["item_id"].values, data["tag_id"].values))).tolil()
    #m = data_sparse.sum()/float(data_sparse.getnnz())
    #data_sparse.data = [elt - m for elt in data_sparse.data]
    print("..........sparse matrix built")

    p, d, q = splin.svds(data_sparse.tocsc(), k)

    # test unitaire sur un item (entrer vrai item id a la main)
    item = 10
    #indices of nonzero tags for this item
    tag_indices = data_sparse.getrow(reinditem[item]).rows[0]
    real_val = data_sparse.getrow(reinditem[item]).data[0]
    print(real_val)
    predsvd = np.dot(np.dot(p,np.diag(d)),q)
    pred = predsvd[reinditem[item]][tag_indices]
    test = (zip(*sorted(zip(real_val, pred, tag_indices))))
    for i in range(len(test[0])):
        print(test[0][i],test[1][i],reindtag[reindtag==test[2][i]].index[0])
    #print(np.std(data_sparse.A),np.std(predsvd),np.std(predsvd-data_sparse.A))

    return p, d, q, reinditem


## INPUT !!!!
####################
p, d, q, reinditem = returnsvd("tags.dat", 50)
f = open("movieLens-10M.tags.dat", "w")




toprint = np.dot(p,np.sqrt(np.diag(d)))
newdata = [(item,toprint[reinditem[item]]) for item in reinditem.index]
np.set_printoptions(suppress=True)
for item in newdata:
    f.write(str(item[0])+"::"+re.sub('[\[\]]', '', np.array2string(item[1],separator=",")).replace(" ","").replace("\n","") + "\n")
f.close()
