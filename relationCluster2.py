__author__ = 'chensn'

import os
import numpy as np
import sktensor

import sklearn.cluster as skclus
import argparse
import time

def supplement(params):
        assert isinstance(params,dict)
        return lambda k,v: params[k] if k in params else v
class TensorDecompAdapter:
     @classmethod
     def cp_als(cls,dt,rank,**params):
        return sktensor.cp_als(dt,rank,**params)
     @classmethod
     def turkey(cls,dt,rank,**params):
         NotImplementedError

class TensorCluster:
    dm = dict(cp_als = 'cp_als',none=None)
    cm = dict(kmeans = 'KMeans',
              miniBatchKmeans = 'MiniBatchKMeans')

    @property
    def tensor3(self):
        return self.__t3

    @tensor3.setter
    def tensor3(self,t3):
        if not isinstance(t3,np.ndarray):
            raise ValueError('Expected object of numpy.ndarray, got {}'.format(type(t3).__name__))
        ndim = len(t3.shape)
        if ndim != 3:
            raise ValueError('Expected 3-order tensor, got {}-order'.format(ndim))
        self.__t3 = t3

    def __init__(self,n_cluster,tensor3,rels,dec,clu,decparams,cluparams,result_dir):

        self.n_cluster = n_cluster
        self.tensor3 = tensor3
        self.rels = rels
        self.dec_method = dec
        self.clu_method = clu
        self.decompo_param(**decparams)
        self.cluster_param(**cluparams)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        self.result_dir = result_dir


    __decompo_param = {}

    def decompo_param(self,**params):
        default = supplement(params)
        params['max_iter'] = default('max_iter',100)
        self.__decompo_param = params


    __cluster_param = {}

    def cluster_param(self,**params):
        default = supplement(params)
        params['n_jobs'] = default('n_jobs',-2)
        params['max_iter'] = default('max_iter',100)
        params['n_clusters'] = self.n_cluster
        self.__cluster_param = params

    def decompostion(self,params):
        t0 = time.time()
        print 'start tensor decomposing'
        self.rank = params.pop('rank',10)
        print "{0}\ndecomposition parameters\n{0}".format('='*len('decomposition parameters'))
        for k,v in params.items():
            print k,'=',v
        dt = sktensor.dtensor(self.tensor3)
        method = getattr(TensorDecompAdapter,self.dec_method)
        P,fit,itr,exectimes = method(dt,self.rank,**params)
        if isinstance(P,sktensor.ktensor):
            P = P.toarray()
        print 'It costs {} s to decompose the tensor'.format((time.time()-t0))
        return P,fit,itr,exectimes
        # P,fit,itr,exectimes  = cp_als(dt,rank,init='random')

    def cluster(self,t3,params):
        t0 = time.time()
        print 'start clustering'
        sz = t3.shape
        t3 = t3.reshape(sz[0]*sz[1],sz[2]).T
        cluster = getattr(skclus,self.clu_method)(**params)
        cluster.fit(t3)
        print 'It costs {:0.2f} s to cluster'.format((time.time()-t0))
        return cluster




    def save_cluster_result(self,result):
        t0 = time.time()
        clusters = [[] for i in range(self.n_cluster)]
        for i,lab in enumerate(result.labels_):
            clusters[lab].append(self.rels[i])
        print "save results to file"
        result_dir = os.path.join(self.result_dir,'clusterResult')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for i in range(self.n_cluster):
            with open(os.path.join(result_dir,'c'+str(i)),'w') as fid:
                for w in clusters[i]:
                    fid.write(w+"\n")
        print "It costs {:0.2f} s to save results ".format((time.time()-t0))

    def save_tc_result(self,t3):
        path = os.path.join(self.result_dir,'tc{}'.format(self.rank))
        np.save(path,t3)

    @property
    def quick(self):
        return False if self.dec_method else True

    def run(self,save_tc_res=False):
        if self.quick:
            self.quick_run()
            return
        print "start running"
        P,fit,itr,exectimes = self.decompostion(self.__decompo_param)
        print "fit = {}\n itr = {}\n execute time = {} s\n".format(fit,itr,exectimes)
        if save_tc_res:
            self.save_tc_result(P)
        result = self.cluster(P,self.__cluster_param)
        self.save_cluster_result(result)

    '''
    clustering relations without tensor decomposition
    '''
    def quick_run(self):
        print "start running"
        result = self.cluster(self.tensor3,self.__cluster_param)
        self.save_cluster_result(result)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('-i','--input',help="path of tensor file")
    parse.add_argument('-r','--rels',help="path of relations file")
    # parse.add_argument('-o','--output',help="directory to store all results")
    args = parse.parse_args()
    t3 = np.load(args.input)
    with open(args.rels) as fid:
        rels = [r for r in fid]

    tc1 = TensorCluster(50,t3,rels,
                        TensorCluster.dm['cp_als'],
                        TensorCluster.cm['kmeans'],
                        dict(rank=10,init='random'),
                        dict(precompute_distances=True,verbose=0),
                        'cp10_kmeans50')
    result = tc1.run(save_tc_res=True)

    # tc2 = TensorCluster(50,t3,rels,
    #                     TensorCluster.dm['none'],
    #                     TensorCluster.cm['kmeans'],
    #                     dict(rank=10,init='random'),
    #                     dict(precompute_distances=True,verbose=0),
    #                     'cp10_kmeans50')
    #
    # tc2.run()
