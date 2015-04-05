__author__ = 'chenshini'

import re
import argparse
import os
import numpy as np
import time
test_path = "D:\\Study\\kb\\data\\dbpedia-mini\\"

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(levelname)s %(message)s')
def select_features(path):
    pro_dict = {}
    rel_dict = {}
    p1 = re.compile(r"(?P<e1><[^>]+>)[ ]+(?P<r><[^>]+>)[ ]+(?P<e2><[^>]+>)")
    p2 = re.compile(r"(?P<e><[^>]+>)[ ]+(?P<p><[^>]+>)[ ]+(?P<v>\"[^\"]+\")")
    fid = open(path)
    for line in fid:
        m = p1.match(line)
        if m != None:
            e1 = m.group("e1")
            r = m.group("r")
            e2 = m.group("e2")
            if r not in rel_dict:
                rel_dict[r] = set([e2])
            else:
                rel_dict[r].add(e2)
            # print e1,r,e2
        else:
            m = p2.match(line)

            if m != None:
                e = m.group("e")
                p = m.group("p")
                v = m.group("v")
                if p not in pro_dict:
                    pro_dict[p] = set([v])
                else:
                    pro_dict[p].add(v)
                # print e,p,v
            else:
                print "Cannot parse the line :",line
    fid.close()
    print "info of relation dict"
    for k,v in rel_dict.items():
        print k,v.__len__()
        # for ele in v:
        #     print ele
    print "info of property dict"
    for k,v in pro_dict.items():
        print k,v.__len__()

    return rel_dict,pro_dict

def isRelation(line):
    p1 = re.compile(r"(?P<e1><[^>]+>)[ ]+(?P<r><[^>]+>)[ ]+(?P<e2><[^>]+>)")
    m = p1.match(line)
    if m != None:
        return True,m.group("r")
    else:
        return False,None
def isProperty(line):
    p2 = re.compile(r"(?P<e><[^>]+>)[ ]+(?P<p><[^>]+>)[ ]+(?P<v>\"[^\"]+\")")
    m = p2.match(line)
    if m != None:
        return True, m.group("p")
    else:
        return False,None
def numOfRelationsAndProperties(paths):

    relations = set([])
    properties = set([])
    for p in paths:
        fid = open(p)
        for line in fid:
            res,rel = isRelation(line)
            if res:
                relations.add(rel)
            else:
                res,prop = isProperty(line)
                if res:
                    properties.add(prop)
        fid.close()
    return relations.__len__(),properties.__len__()

def readTriple(path,kb,isEnts,isRels):
    if not isinstance(kb,list):
        raise ValueError('kb must be list')
    if not isinstance(isEnts,dict):
        raise  ValueError('isEnts must be a dict')
    if not isinstance(isRels,dict):
        raise  ValueError('isRels must be a dict')
    fid = open(path)
    p = re.compile(r"(?P<e1><[^>]+>)[ ]+(?P<r><[^>]+>)[ ]+(?P<e2>(<[^>]+>)|(\"[^\"]+\"))")
    # tuples = []
    for line in fid:
        m = p.match(line)
        if m == None:
            continue

        e1 = m.group("e1")
        r = m.group("r")
        e2 = m.group("e2")
        isEnts[e1] =1
        if not e2.startswith("\""):
            isRels[r] = 1
            isEnts[e2] = 1
        kb.append(tuple((e1,r,e2)))
    return kb,isEnts,isRels

def loadKBFromMulFile(paths):
    kb = []
    isEnts = {}
    isRels = {}
    for p in paths:
        print "load data from %s" % p
        kb,isEnts,isRels = readTriple(p,kb,isEnts,isRels)
    return kb,isEnts,isRels

def getPointsWithLinks(kb,isEnts):
    # kb = readTripFromMulFile(paths)
    innerPoints = {}
    for t in kb:
        if t[0] not in innerPoints:
            innerPoints[t[0]] = set([])
        if isEnts.has_key(t[2]) and t[2] not in innerPoints:
            innerPoints[t[2]] = set([])

        innerPoints[t[0]].add(t[1])
        if isEnts.has_key(t[2]):
            innerPoints[t[2]].add(t[1])
    return innerPoints

def uniqify(seq):
    # Not order preserving
    return {}.fromkeys(seq).keys()
'''
build feature matrix for relations regardless of direction
'''
def kb2tensor3(kb,isEnts,isRels):
    plinks = getPointsWithLinks(kb,isEnts)
    rels_props = uniqify([t[1] for t in kb])
    indices3 = {}
    indices12 = {}
    for i, r in enumerate(isRels.keys()):
        indices3[r] = i
    for i,r in enumerate(rels_props):
        indices12[r] = i
    dim12 = len(indices12)
    dim3 = len(indices3)
    print dim12,dim3
    tensor3 = np.zeros((dim12,dim12,dim3))
    for t in kb:
        rel = t[1]
        if not isRels.has_key(rel):
            continue
        # mat =
        for le in plinks[t[0]]:
            if le == rel:
                continue
            li = indices12[le]
            for re in plinks[t[2]]:
                if re == rel:
                    continue
                ri = indices12[re]
                tensor3[li,ri][indices3[rel]]+=1
    print "complete tensor3 construction"
    return tensor3,indices12,indices3
def allPaths(dir):
    return [os.path.join(dir,p) for p in os.listdir(dir)]
def printTensor(tensor3):
    sz = tensor3.shape
    for k in range(sz[2]):
        for i in range(sz[0]):
            for j in range(sz[1]):
                print tensor3[i,j,k],
            print
        print
def saveTensor3(path,tensor3):
    out = open(path,'w')
    sz = tensor3.shape
    for k in range(sz[2]):
        for i in range(sz[0]):
            for j in range(sz[1]):
                out.write("%d " % tensor3[i,j,k])
            out.write('\n')
        out.write('\n')
    out.close()



def readTriple2(path,ere,epv):
    fid = open(path)
    p = re.compile(r"(?P<e1><[^>]+>)[ ]+(?P<r><[^>]+>)[ ]+(?P<e2>(<[^>]+>)|(\"[^\"]+\"))")
    # tuples = []
    for line in fid:
        m = p.match(line)
        if m == None:
            continue
        e1 = m.group("e1")
        r = m.group("r")
        e2 = m.group("e2")
        if not e2.startswith("\""):
            ere.append(tuple((e1,r,e2)))
        else:
            epv.append(tuple((e1,r,e2)))
    fid.close()
    return ere,epv
def zipLinks2(ere,epv):
    elinks = {}
    for t in ere:
        if not elinks.has_key(t[0]):
            elinks[t[0]] = set([])
        if not elinks.has_key(t[2]):
            elinks[t[2]] = set([])
        elinks[t[0]].add(t[1])
        elinks[t[2]].add(t[1])
    for t in epv:
        if not elinks.has_key(t[0]):
            elinks[t[0]] = set([])
        elinks[t[0]].add(t[1])
    return elinks
def loadKBFromMulFile2(paths):
    if not isinstance(paths,list):
        raise TypeError("paths must be list")
    ere = []
    epv = []
    for p in paths:
        print "load data from %s" % p
        ere,epv = readTriple2(p,ere,epv)
    return ere,epv

def kb2tensor3llp(ere,epv):
    print "build matrix for each relations"
    elinks = zipLinks2(ere,epv)
    rels = uniqify([t[1] for t in ere])
    props = uniqify([t[1] for t in epv])
    ridx = {}
    pidx = {}
    for i,r in enumerate(rels):
        ridx[r] = i
    for i,p in enumerate(props):
        pidx[p] = i
    dim3 = len(rels)
    dim1 = len(props)+dim3
    print "tensor's shape is (%d %d %d)" % (dim1,dim1,dim3)
    tensor3 = np.zeros((dim1,dim1,dim3))
    for t in ere:
        rel = t[1]
        i3 = ridx[rel]
        e1 = t[0]
        e2 = t[2]
        indexof = lambda r: ridx[r] if ridx.has_key(r) else dim3 + pidx[r]
        for ll in elinks[e1]:
            i1 = indexof(ll)
            for rl in elinks[e2]:
                i2 = indexof(rl)
                tensor3[i1,i2,i3] += 1
    return tensor3,rels,props
def loadKbAsTensor(paths):
    ere, epv = loadKBFromMulFile2(paths)
    tensor3,rels,props =kb2tensor3llp(ere,epv)
    return tensor3,rels,props
def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("-i","--input",help="path of the source knowledge base")
    parse.add_argument("-o","--output",help="output file path")
    parse.add_argument("-f","--func",help="choice of function:[1:transform kb to tensor | 2: extract relations and properties]")
    parse.add_argument("-r","--rels",help="path of file to store relations")
    parse.add_argument("-p","--props",help="path of file to store props")
    args = parse.parse_args()
    input = args.input
    if os.path.isfile(input):
        paths = [input]
    else:
        paths = allPaths(input)
    output = args.output
    relpath = args.rels
    propspath = args.props
    if args.func == "1":
        tensor3,rels,props = loadKbAsTensor(paths)
        print "save tensor3 to '%s'" % output
        np.save(output,tensor3)
        print "sava relation list"
        with open(relpath,'w') as rel_out:
            for r in rels:
                rel_out.write("%s\n" % r)
        with open(propspath,'w') as prop_out:
            for p in props:
                prop_out.write("%s\n" % p)
    elif args.func == "2":
        ere,epv = loadKBFromMulFile2(allPaths(args.input))
        t0 =  time.time()
        print "only remain unique relations"
        rels = uniqify([t[1] for t in ere])
        print "the process costs %f " % (time.time()-t0)
        t1 = time.time()
        print "only remain unique properties"
        props = uniqify([t[1] for t in epv])
        print "the process costs %f " % (time.time()-t1)
        print "sava relation list"
        rel_out = open('rels.txt','w')
        for r in rels:
            rel_out.write("%s\n" % r)
        rel_out.close()
        prop_out = open("props.txt",'w')
        for p in props:
            prop_out.write("%s\n" % p)
        prop_out.close()

main()
    # paths =["D:\\Source Code\RelationCluster\\test.txt"]
    # paths = [test_path+"xaa"]
    # paths = allPaths(test_path)
    # kb, isEnts, isRels = loadKBFromMulFile(paths)
    # tensor3, indices12,indices3= kb2tensor3(kb,isEnts,isRels)

    # P,fit,itr,exectimes  = cp_als(dt,10,init='random')
    # print 10,fit
    # P,fit,itr,exectimes  = cp_als(dt,50,init='random')
    # print 50,fit
    # P,fit,itr,exectimes  = cp_als(dt,100,init='random')
    # print 100,fit
    # P,fit,itr,exectimes  = cp_als(dt,500,init='random')
    # print 500,fit
# main()
# select_features(test_path+"/xaa")
# kbSlices = [os.path.join(test_path,p) for p in os.listdir(test_path)]
# print numOfRelationsAndProperties(kbSlices)
