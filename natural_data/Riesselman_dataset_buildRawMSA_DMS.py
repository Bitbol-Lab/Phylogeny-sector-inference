#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construct MSAs and DMS data from datasets in
Adam J. Riesselman, John B. Ingraham, and Debora S. Marks. Deep generative models of genetic
variation capture the effects of mutations. Nature Methods, 15(10):816–822, Oct 2018.

DMS with real data from Riesselman dataset
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import scipy.io
import random
from Bio import SeqIO
from numba import jit
import pandas as pd
from collections import Counter
import matplotlib.colors as mcolors
from scipy.stats import entropy
import scipy.stats
import sklearn
import math
import multiprocessing as mp
from joblib import Parallel, delayed
import swalign


def ReadExcel(dictionary,key):
    ##do a dictionary with entries for the sheetname, idx for fitness
    sheetname = key
    filepath = './41592_2018_138_MOESM4_ESM(1).xls'
    df = pd.read_excel(filepath, sheet_name=sheetname)
    list_keys = df.keys()

    mutants = np.array(df['mutant'])
    str_muteffect = dictionary[key]['fitnesscol']
    dms_values = np.array(df[str_muteffect])

    return mutants,dms_values


def ConstructMatrixDMS(dms_values,mutants,alphabet):
    
    total_length = mutants.shape[0]
    startidx = int(mutants[0][1:-1])
    length_prot = int(mutants[-1][1:-1])-startidx+1
    dms = np.zeros((len(alphabet),length_prot))
    dms[:] = np.nan ###test
    sequence_ref = np.empty(length_prot,dtype = '<U1')
    pos_prev = startidx
    idxs = [] 
    for idxm,mut in enumerate(mutants):
        letref = mut[0]
        letmutation = mut[-1]
        pos = int(mut[1:-1])-startidx
        sequence_ref[pos] = letref
        if letmutation in alphabet:
            dms[alphabet.index(letmutation),pos] = dms_values[idxm]
        if pos-pos_prev  > 1:
            nbrpos = pos-pos_prev
            for i in range(1,nbrpos):
                idxs.append(pos_prev+i)
                if pos_prev + i == length_prot:
                    import pdb;pdb.set_trace()
        pos_prev = pos
    dms_new = np.delete(dms,idxs,axis = 1)
    return dms_new,sequence_ref


def OpenFasta(path_seq):
    records = list(SeqIO.parse(path_seq, "fasta"))
    sequences = []
    headers = []

    for r in records:
        sequences.append(r.seq)
        headers.append(r.id)
    return np.array(sequences), headers

def OpenFastaList(path_seq):
    records = list(SeqIO.parse(path_seq, "fasta"))
    sequences = []
    headers = []

    for r in records:
        sequences.append(r.seq)
        headers.append(r.id)
    return sequences, headers

def KeepMatchStatesRewriteFastafile(path, dest_fasta_path):
    sequences,headers = OpenFastaList(path)
    
    with open(dest_fasta_path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(">"+headers[i]+"\n")
            strtmp = ''.join([x for x in seq if not(x.islower())])
            f.write(strtmp+"\n")
    
def RemoveColumnGaps(fastapath):
    sequences,headers = OpenFasta(fastapath)
    nbrseq,nbrpos = sequences.shape
    idx_keep = []
    for p in range(0,nbrpos):
        val,cts = np.unique(sequences[:,p],return_counts = True)
        if '-' in val:
            if cts[val == '-']/nbrseq <= 0.3:
                idx_keep.append(p)
        else:
            idx_keep.append(p)
    return sequences[:,idx_keep],headers

def WriteFastaFile(sequences,headers, dest_fasta_path):
    # sequences,headers = OpenFasta(path)
    
    with open(dest_fasta_path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(">"+headers[i]+"\n")
            strtmp = ''.join([x for x in seq])
            f.write(strtmp+"\n")          

def FilterMSACoverage(sequences,headers):

    nbrseq, length= sequences.shape
    idxs_keep = []
    for idxseq in range(0,nbrseq):
        val,cts = np.unique(sequences[idxseq,:],return_counts =True)
        if '-' in val:
            nbrgaps = cts[val == '-']
            if (length-nbrgaps)/length > 0.8:
                idxs_keep.append(idxseq)
        else:
            idxs_keep.append(idxseq)
            

    headers = np.array(headers)
    return sequences[idxs_keep,:], headers[idxs_keep]

def Extractinfos_xcl(path,keysheet,listdone_fam = []):
    df =pd.read_excel(path, sheet_name=keysheet)
    protfamily = []
    datasetnames = []
    colname_dmsval =[]
    fitnessnamecolstr = list(df['Name(s) in Reference'])
    protnames = list(df['Protein name'])
    dtsetnames = list(df['Dataset Name(s)'])
    
    fig3col = list(df['In Figure 3'])
    keysheetnames = list(df['Keys dataset sheet'])
    dictionary = {}
    for idx,key in enumerate(keysheetnames):
        if isinstance(key, str) and not(key in listdone_fam):
            protfamily.append(protnames[idx])
            colname_dmsval.append(fitnessnamecolstr[idx])
            datasetnames.append(key)
            dictionary[key] = {'fitnesscol':str(fitnessnamecolstr[idx]),'Protein':protnames[idx]}
    return protfamily,datasetnames,colname_dmsval,dictionary

def Task(key,mainpath, dictionary,alphabet,pthuniref):
    path_onefam = mainpath+'/'+str(key)
    os.mkdir(path_onefam)
    Startjackhammer(dictionary, key, path_onefam, alphabet)

    pathmsa = Launchjackhmmer(pthuniref, key, path_onefam)
    ProcessJackhmmerMSA(pathmsa,path_onefam)

def Startjackhammer(dictionary,key,path,alphabet):
    mutants,fitness = ReadExcel(dictionary,key)

    dms,seqref = ConstructMatrixDMS(fitness, mutants, alphabet)

    np.save(path+'/dms_raw.npy',dms)
    
    with open(path+'/refseq_'+key, "w") as f:
            f.write(">"+'Reference_sequence'+"\n")
            strtmp = ''.join(seqref)
            f.write(strtmp+"\n")

    np.save(path+'/seqreflength',seqref.shape[0])

       
def Launchjackhmmer(pathuniref,key,savepath):
    length = int(np.load(savepath+'/seqreflength.npy'))
    pathrawref = savepath+'/refseq_'+key
    bitscore = length*0.5
    nbrcpus = 6
    nbriter = 5
    if not(os.path.exists(savepath+'/MSA_raw_jackhmmr')):
        os.mkdir(savepath+'/MSA_raw_jackhmmr')
        bitscorename = str(bitscore).replace('.','_')
        storemsapath = savepath+'/MSA_raw_jackhmmr/MSAstock_jackhmmer{}'.format(bitscorename)
        
        launchstr = 'jackhmmer --incdomT {} --cpu {} -N {} -A {} {} {}'.format(bitscore,nbrcpus,nbriter,storemsapath,pathrawref,pathuniref)
        os.system(launchstr)
        # jackhmmer --incdomT 131.5 --cpu 6 -N 5 -A ./BetaLactamase/pf144_jackhmmersearchMSAstock131_5 ./BetaLactamase/Jackhmmer_msa/query_sequence_referenceDMS_BetaLactamase.fasta ./Uniref100/uniref100.fasta
    else:
        print('Jackhmmer MSA already exists')
        bitscorename = str(bitscore).replace('.','_')
        storemsapath = savepath+'/MSA_raw_jackhmmr/MSAstock_jackhmmer{}'.format(bitscorename)

    return storemsapath

def ProcessJackhmmerMSA(pathmsa,savepath):
    newpath = savepath+'/ProcessedMSA'
    if not(os.path.exists(newpath)):
        os.mkdir(newpath)
    msaname = pathmsa.split('/')[-1]
    savepath_nw = newpath+'/'+msaname+'_a2m'
    eslstr = 'esl-reformat -o {} a2m {}'.format(savepath_nw,pathmsa)
    os.system(eslstr)
    path_KM = savepath_nw+'_KEEPMST'
    KeepMatchStatesRewriteFastafile(savepath_nw, path_KM)
    sequences,headers =  RemoveColumnGaps(path_KM)
    newsequences,newheaders = FilterMSACoverage(sequences,headers)
    newseqflip = np.flip(newsequences,axis = 0)
    newheadersflip = np.flip(newheaders)
    WriteFastaFile(newseqflip,newheadersflip, path_KM+'_RM30fgaps_filter80cov')


def AlignRefQuery(ref,query, famname):
    match = 2
    mismatch = -1
    scoring = swalign.IdentityScoringMatrix(match, mismatch)
    sw = swalign.LocalAlignment(scoring, gap_penalty=0)
    alignment = sw.align(ref, query)
    
    if alignment.q_pos != 0:
        print('warning query not 0 start '+ famname)
    startidx = alignment.r_pos
    idxs_del = []
    idxs_del = idxs_del + list(np.arange(startidx))

    counting_idxs = alignment.r_pos-1

    cigar = alignment.cigar

    for el in cigar:
        nbrsites,state = el
        if state == 'M':
            counting_idxs = counting_idxs + nbrsites
        elif state =='D':
            for i in range(1,nbrsites+1):
                counting_idxs = counting_idxs + 1
                idxs_del.append(counting_idxs)

        elif state != 'M' and state != 'D':
            print('not M or D state: '+ state +' '+famname)

    total_size = len(ref)
    endidx = alignment.r_end

    if counting_idxs < total_size-1:
        assert endidx == counting_idxs+1
        idxs_del = idxs_del + list(np.arange(endidx,total_size,1))
        counting_idxs =counting_idxs+ len(list(np.arange(endidx,total_size,1)))

    assert counting_idxs+1 == total_size

    return idxs_del
        
def RemoveFilesDot(listdot):
    for f in listdot:
        if f.startswith('.'):
            listdot.remove(f)
    return listdot


def AlignSequences(path_main):
    listfolders = RemoveFilesDot(os.listdir(path_main))
    for folder in listfolders:
            # print(folder)
        path_family =path_main +'/'+ folder
        seqrefname = 'refseq_'+folder
        path_ref = path_family +'/'+seqrefname
        msa_path = path_family+'/ProcessedMSA/'
        for f in RemoveFilesDot(os.listdir(msa_path)):
            if '_KEEPMST_RM30fgaps_filter80cov' in f:
                msa_pathfinal = msa_path+f
        msa,_ = OpenFasta(msa_pathfinal)
        query = ''.join(msa[0,:])
        refseq,_ = OpenFasta(path_ref)
        refseq = refseq[0]
        ref = ''.join(refseq)
        
        
        nbrseq, nbrpos = msa.shape
        if nbrseq < 10*nbrpos:
            print('warning not enough sequences for family '+folder)
        
        idxsdel = AlignRefQuery(ref,query,folder)
        
        np.save(path_family+'/'+'idxs_del.npy',idxsdel)
        # dmsraw = np.load(path_family+'/dms_raw.npy')
        
        dmsraw = np.load(path_family+'/dms_raw.npy')
        
        aligned_dms = np.delete(dmsraw,idxsdel,axis = 1)
        aligned_refseq = np.delete(refseq,idxsdel)
        
        np.save(path_family+'/dms_aligned.npy',aligned_dms)
        WriteFastaFile([aligned_refseq],['Reference_sequence'], path_family+'/reference_sequence_aligned')
     
        
if __name__ == '__main__':
        
    alphabet = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    
    pthmain = './MSAs'
    
    #FILL PATH TO UNIREF
    pthuniref = './uniref100.fasta'
    #listdone = os.listdir(pthmain)
    
    _,_,_,dictionary = Extractinfos_xcl('./Metadata_dms_keysheetnames.xls','data')
    
    if mp.get_start_method() == "spawn":
        import sys
        sys.exit(0)
    backend = 'multiprocessing'
    
    Parallel(n_jobs=1)(delayed(Task)(key,pthmain, dictionary,alphabet,pthuniref) for key in dictionary)
    AlignSequences('./MSAs')
