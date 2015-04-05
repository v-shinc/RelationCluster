
__author__ = 'chensn'

import numpy as np
from sktensor import dtensor,cp_als
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import argparse
import os
def rel_cluster(tensor3,rank,n_clusters):

    dt = dtensor(tensor3)
    print "tensor decomposition"
    P,fit,itr,exectimes  = cp_als(dt,rank,init='random')
    print rank,fit
    print "compute relation clustering with Means"
    tensor3 = P.toarray()
    sz = tensor3.shape
    print sz
    # one data sample for each row
    tensor3 = tensor3.reshape(sz[0]*sz[1],sz[2])
    tensor3 = tensor3.T
    k_means = KMeans(init='k-means++',n_clusters=n_clusters,n_init=3,n_jobs=-1)
    k_means.fit(tensor3)
    return k_means

def test():
    centers = [[1, 1], [-1, -1], [1, -1]]
    n_clusters = len(centers)
    X, labels_true = make_blobs(n_samples=10, centers=centers, cluster_std=0.7)


    t3 = np.asarray([[1, 1], [-1, -1], [1, -1],[1,1],[-1,-1]])
    print t3
    k_means = KMeans(init='k-means++',n_clusters=3,n_init=3)
    k_means.fit(t3)
    print k_means.labels_
    # k_means = KMeans(init='k-means++', n_clusters=3, n_init=3)
    # k_means.fit(t3)
    # print k_means.labels_

'''file format: one word for each line'''
def loadListFile(path):
    fid = open(path)
    l = [line for line in fid]
    fid.close()
    return l

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("-i","--input",help = "path of tensor file")
    parse.add_argument("-r","--rels",help="path of relations file")
    parse.add_argument("-p","--props",help="path of properties file")
    parse.add_argument("-o","--output",help = "path directory to save clustering result")
    args = parse.parse_args()
    input = args.input

    tensor3 = np.load(input)
    print "load relations"
    rels = loadListFile(args.rels)


    print "cp decomposition and cluster relations using kmeans"
    n_cluster = 20
    result = rel_cluster(tensor3,10,n_cluster)
    # initial clusters
    clusters = [[] for i in range(n_cluster)]
    for i,lab in enumerate(result.labels_):
        clusters[lab].append(rels[i])
    print "save results to file"
    outdir = args.output
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for i in range(n_cluster):
        fid = open(os.path.join(outdir,'c'+str(i)),'w')
        for w in clusters[i]:
            fid.write(w+"\n")
        fid.close()
main()



