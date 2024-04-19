#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construct MSA and do ICOD,MI,SCA
##convert stockholm to fasta : esl-reformat -o PF71_full_fasta_test a2m PF00071_alignment_full
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
#import utils as u
import os
from tqdm import tqdm
import scipy.io
import random
from Bio import SeqIO
from numba import jit
from numba.typed import List
import multiprocessing as mp
from joblib import Parallel, delayed
import numba as nb
import pandas as pd
from collections import Counter
from scipy.stats import entropy
import swalign
nb.set_num_threads(3)
import re
import time
from scipy.sparse import csr_matrix as sparsify
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cluster import AgglomerativeClustering



###############################################################################    
##  PREPARE MSAs
###############################################################################    
  
def LoadSequences(path_seq):
    records = list(SeqIO.parse(path_seq, "fasta"))
    # import pdb;pdb.set_trace()
    sequences = []
    headers = []
    # import pdb;pdb.set_trace()
    for r in records:
        sequences.append(r.seq)
        headers.append(r.id)
    return sequences, headers

@jit(nopython = True)
def ComputeHammTOJKCArray(inputarr,nbrpos,nbrstates):
    p_array = inputarr/nbrpos
    thr = 0.949*np.ones(p_array.shape)
    p_array = np.minimum(p_array,thr)
    return -(nbrstates-1)/nbrstates*np.log(1-nbrstates*p_array/(nbrstates-1))

@nb.njit(parallel=True)
def ReplaceGaps(msa, jcmat):
    nbrseq, nbrpos = msa.shape
    
    
    for s in nb.prange(nbrseq):
        seqtmp = msa[s,:].copy()
        
        for p in range(0,nbrpos):
            tmpvec = ComputeHammTOJKCArray(jcmat[s,:].copy(),nbrpos,20)
            if seqtmp[p] == 0:
                idx_nearest_seq = np.argmin(tmpvec)
                while(msa[idx_nearest_seq,p] == 0):
                    tmpvec[idx_nearest_seq] = 100
                    idx_nearest_seq = np.argmin(tmpvec)
                msa[s,p] = msa[idx_nearest_seq,p]
    
    return msa

@nb.njit(parallel=True)
def HammingDistanceMatrix(mat,dt):
    depth,_ = mat.shape
    #ham = np.empty((depth * (depth - 1)) // 2, dtype=np.uint8)
    jcmat =np.zeros((depth,depth),dtype = dt)
    for i in nb.prange(len(mat)):
        for j in range(i + 1, len(mat)):
            jcmat[i,j] = (mat[i] != mat[j]).sum()
    return jcmat+jcmat.T


def ConvertStrToInt(msa):

    l = list(np.unique(sequences))
    # l = ['-','A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    nbrseq, nbrpos = msa.shape
    msanumb = np.zeros(msa.shape, dtype = np.int8)
    
    for s in range(nbrseq):
        for p in range(nbrpos):
            msanumb[s,p] = l.index(msa[s,p])
    
    return msanumb

def ConvertFromAAToints(matrix):
    aalist = list(np.unique(matrix))
    # import pdb;pdb.set_trace()
    if 'X' in aalist:
        aalist.remove('X')
    if 'B' in aalist:
        aalist.remove('B')
    if 'Z' in aalist:
        aalist.remove('Z')     
        
    nbrseq,nbrpos = matrix.shape
    newarray = np.empty((nbrseq,nbrpos), dtype = np.int8)
    for i in range(0,nbrseq):
        for j in range(0,nbrpos):
            if matrix[i,j] not in  ['X','B','Z']:
                newarray[i,j] = aalist.index(matrix[i,j])
            else:
                newarray[i,j] = aalist.index('-')
    return newarray,aalist



def JukesCantor(refseq, msa, nbrstates):
    nbrseq, nbrpos = msa.shape
    distances = np.zeros(nbrseq)
    for idxs,s in enumerate(range(0,nbrseq)):
        seqtmp = msa[s,:]
        ct = 0
        for idxaa, aa in enumerate(refseq):
            if aa != seqtmp[idxaa]:
                ct += 1
        p = ct/len(refseq)
        p = min(p,.949)
        
        distances[idxs] = -(nbrstates-1)/nbrstates * np.log(1 - nbrstates*p/(nbrstates-1))


    return distances

def PrepareMSA(pth,refseq_id,jcmax):

    seq, headers = LoadSequences(pth)
    sequences = np.array(seq)
    sequence_alignment = sequences[headers.index(refseq_id),:].copy()
    sequences = sequences[:,sequence_alignment != '-']

    sequences_sorted, sorted_dist = Phylosort(sequences,headers.index(refseq_id))

    if sorted_dist[-1] >= jcmax:
        idxs = np.argwhere(sorted_dist>=jcmax)

        sequences_keep = sequences_sorted[:idxs[0][0]+1,:]
        sorted_dist_newmsa = sorted_dist[:idxs[0][0]+1]
    else:
        sequences_keep = sequences_sorted
        sorted_dist_newmsa = sorted_dist
    

    msaints,aalist = ConvertFromAAToints(sequences_keep)
    nbrseq,nbrpos = msaints.shape
    
    if nbrpos < 256:
        dt = np.uint8
    else:
        dt = np.uint16
        
    pairwised = HammingDistanceMatrix(msaints,dt)

    finalmsa = ReplaceGaps(msaints, pairwised)

    
    return finalmsa

def Phylosort(sequences,refseqidx):

    refseq = sequences[refseqidx,:]
    
    distances = JukesCantor(refseq, sequences,20) 
    distances[-1] = 0  #following code of matlab -> of article Extracting phylogenetic dimensions of coevolution reveals hidden functional signals , Colavin 2022
    sorted_idxs = np.argsort(distances)
    sorted_msa = sequences[sorted_idxs,:]
    
    return sorted_msa, distances[sorted_idxs]

def MSA_Cutoffs(pathfolder,folder,refseqDMS_id):
    pathtmp = path_folder+folder +'/ProcessedMSA/'
    for f in os.listdir(pathtmp):
        if '_KEEPMST_RM30fgaps_filter80cov' in f:
            fullpath = pathtmp+f
    
    cutoffs = [0.4,0.6,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    
    pathsaveresampled = path_folder+folder +'/FinalMSAs/cutoffs/'
    if not(os.path.exists(pathsaveresampled)):
        os.mkdir(path_folder+folder +'/FinalMSAs/')
        os.mkdir(pathsaveresampled)
    print(folder)
    for c in tqdm(cutoffs):
        if not(os.path.exists(pathsaveresampled+str(c).replace('.','_'))):

            finalmsa =  PrepareMSA(fullpath, refseqDMS_id, c)
            finalmsa = finalmsa-1

            nwpath = pathsaveresampled+str(c).replace('.','_')
            os.mkdir(nwpath)
            
            np.save(nwpath+'/MSA_cutoff'+str(c).replace('.','_')+'.npy',finalmsa)


###############################################################################    
##  IMPLEMENTATION OF METHODS
###############################################################################    

def Createpath(path):
    if not(os.path.exists(path)):
        os.mkdir(path)
        
@jit(nopython = True)
def ComputeFrequenciesQstates(msa):
    nbrseq,nbrpos = msa.shape


    binst = np.linspace(0,20,21)
    frequencies = np.zeros((20, nbrpos))

    for p in range(0,nbrpos):

        cts,val = np.histogram(msa[:,p],bins = binst)

        frequencies[:,p] = cts/nbrseq

    return frequencies


def MIMatrix(MSA, nbrstates,p,frequencies):
    nbrseq,nbrpos = MSA.shape
    mi_matrix = np.zeros((nbrpos,nbrpos))
    
    nbrstates -= 1
    binst = np.linspace(0,nbrstates+1, nbrstates+2)
    for i in range(0,nbrpos):
        for j in range(0,nbrpos):
            if i != j:
                fri = p/(nbrstates+1) + (1-p)*frequencies[:,i]
                frj = p/(nbrstates+1) + (1-p)*frequencies[:,j]
                
                hist2d,_,_ = np.histogram2d(MSA[:,i],MSA[:,j],bins = binst) 

                jointfreq = (p/(nbrstates+1)**2) + (1-p)*hist2d/nbrseq
                
                jointentropy = entropy(jointfreq.flatten())
                
                entropy_i = entropy(fri)
                entropy_j = entropy(frj)

                
                try:
                    mi_matrix[i,j] = (entropy_i + entropy_j - jointentropy)/jointentropy
                except RuntimeWarning:

                    mi_matrix[i,j] = 0
               
            else:
                fri = p/(nbrstates+1) + (1-p)*frequencies[:,i]
                submatrix = np.zeros([nbrstates+1, nbrstates+1])
                submatrix[np.diag_indices_from(submatrix)] = fri
                
                jointentropy = entropy(submatrix.flatten())
                
                entropy_i = entropy(fri)
                mi_matrix[i,j] = (entropy_i + entropy_i - jointentropy)/jointentropy

    return mi_matrix


def ComputeNJEInts(sequences, pseudocount):
    frequencies = ComputeFrequenciesQstates(sequences)
    # import pdb;pdb.set_trace()
    mimatrix = MIMatrix(sequences, 20,pseudocount,frequencies)
    
    return mimatrix

def APCNodiagMATLAB(matrix):

    nbr_sites,_ = matrix.shape
    mat_apc = np.zeros(matrix.shape)
    
    matrix_zerodiag= matrix.copy()
    matrix_zerodiag[np.diag_indices_from(matrix_zerodiag)] = 0    
    
    mean_overall = 2/(nbr_sites*(nbr_sites-1))*(np.sum(np.triu(matrix_zerodiag, k = 1)))

    for k in range(0,nbr_sites):
        for i in range(k, nbr_sites):
            m_i = (np.sum(matrix_zerodiag[0:k,i])+ np.sum(matrix_zerodiag[k+1:,i]))/(nbr_sites-1)
            m_k = (np.sum(matrix_zerodiag[k,0:i])+ np.sum(matrix_zerodiag[k,i+1:]))/(nbr_sites-1)
            
            mat_apc[k,i] = matrix[k,i] - m_i*m_k/mean_overall
            mat_apc[i,k] = matrix[i,k] - m_i*m_k/mean_overall

    return mat_apc


@jit(nopython = True)
def ComputeFrequenciesRefSeqNumeric(msa, refseq):
    nbrseq,nbrpos = msa.shape
    l = list(np.unique(msa))
    
    binst = np.linspace(0,len(l),len(l)+1)
    frequencies = np.zeros((len(l)-1, nbrpos))

    for p in range(0,nbrpos):

        cts,val = np.histogram(msa[:,p],bins = binst)
        
        index_skip = refseq[p]
        

        cts = np.delete(cts,index_skip)
        
        frequencies[:,p] = cts/nbrseq

    return frequencies

def CovarianceMatrixPseudocountNumeric(MSA,nbrstates, p, frequencies, refseq):
    nbrseq,nbrpos = MSA.shape
    nbrstates -= 1
    C = np.zeros((nbrpos*nbrstates,nbrpos*nbrstates))

    binst = np.linspace(0,nbrstates+1, nbrstates+2)
    for i in range(0,nbrpos):
        for j in range(0,nbrpos):
            if i != j:
                fri = p/(nbrstates+1) + (1-p)*frequencies[:,i]
                frj = p/(nbrstates+1) + (1-p)*frequencies[:,j]
                
                hist2d,_,_ = np.histogram2d(MSA[:,i],MSA[:,j],bins = binst) 
                # hist2d = hist2d[:-1,:-1]
                
                skip_indexi = refseq[i]
                skip_indexj = refseq[j]
                
                hist2dtmp = np.delete(hist2d,skip_indexi, axis = 0)
                hist2dtmp = np.delete(hist2dtmp, skip_indexj, axis = 1)
                
                # import pdb;pdb.set_trace()

                jointfreq = (p/(nbrstates+1)**2) + (1-p)*hist2dtmp/nbrseq
                C[i*nbrstates:(i+1)*nbrstates, j*nbrstates:(j+1)*nbrstates] = jointfreq - np.tensordot(fri,frj,0)
                # C[j*nbrstates:(j+1)*nbrstates,i*nbrstates:(i+1)*nbrstates] = C[i*nbrstates:(i+1)*nbrstates, j*nbrstates:(j+1)*nbrstates].T
            else:
                fri = p/(nbrstates+1) + (1-p)*frequencies[:,i]
                submatrix = np.zeros([nbrstates, nbrstates])
                submatrix[np.diag_indices_from(submatrix)] = fri
                C[i*nbrstates:(i+1)*nbrstates, j*nbrstates:(j+1)*nbrstates] = submatrix - np.tensordot(fri,fri,0)
                
    return C



@jit(nopython = True)
def CompressICOD(icod,nbrstates,nbrpos):
    nbrstates -= 1
    icod_final = np.zeros((nbrpos,nbrpos))
    for i in range(0,nbrpos):
        for j in range(i+1,nbrpos):
            frobeniusnorm = np.linalg.norm(icod[i*nbrstates:(i+1)*nbrstates, j*nbrstates:(j+1)*nbrstates])
            icod_final[i,j] = frobeniusnorm

            if i != j:
                icod_final[j,i] = frobeniusnorm

    return icod_final


def icod_frobnorm(msa_numeric):
    nbrseq,nbrpos = msa_numeric.shape
    nbraa = len(np.unique(msa_numeric))
    refseq = msa_numeric[0,:].copy()
    frequencies = ComputeFrequenciesRefSeqNumeric(msa_numeric,refseq)
    C = CovarianceMatrixPseudocountNumeric(msa_numeric, nbraa,0.05,frequencies, refseq)
    icodtmp = np.linalg.inv(C)
    icod = CompressICOD(icodtmp, nbraa, nbrpos)
    return icod

def APCNodiagMATLAB(matrix):

    nbr_sites,_ = matrix.shape
    mat_apc = np.zeros(matrix.shape)
    
    matrix_zerodiag= matrix.copy()
    matrix_zerodiag[np.diag_indices_from(matrix_zerodiag)] = 0    
    
    mean_overall = 2/(nbr_sites*(nbr_sites-1))*(np.sum(np.triu(matrix_zerodiag, k = 1)))

    for k in range(0,nbr_sites):
        for i in range(k, nbr_sites):
            m_i = (np.sum(matrix_zerodiag[0:k,i])+ np.sum(matrix_zerodiag[k+1:,i]))/(nbr_sites-1)
            m_k = (np.sum(matrix_zerodiag[k,0:i])+ np.sum(matrix_zerodiag[k,i+1:]))/(nbr_sites-1)
            
            mat_apc[k,i] = matrix[k,i] - m_i*m_k/mean_overall
            mat_apc[i,k] = matrix[i,k] - m_i*m_k/mean_overall

    return mat_apc


def ICOD_MSACutoff(pathmsa,blapc):
    msa = np.load(pathmsa)
    icod_originalmsa = icod_frobnorm(msa)    
    if blapc:
        icod_originalmsa = APCNodiagMATLAB(icod_originalmsa)
    return np.linalg.eigh(icod_originalmsa)

def TaskICOD(path_family,cutoff):
    if not(os.path.exists(path_family+cutoff+'/ICOD')):
        path_c = path_family+cutoff

        msacutoff = np.load(path_c+'/MSA_cutoff'+cutoff+'.npy')
        
        egval_icod,egvect_icod = ICOD_MSACutoff(path_c+'/MSA_cutoff'+cutoff+'.npy', False)
        egval_icod_apc,egvect_icod_apc = ICOD_MSACutoff(path_c+'/MSA_cutoff'+cutoff+'.npy', True)

        os.mkdir(path_c+'/ICOD')
        os.mkdir(path_c+'/ICOD/APC');os.mkdir(path_c+'/ICOD/NO_APC')
        
        np.save(path_c+'/ICOD/NO_APC/eigenvalues.npy',egval_icod)
        np.save(path_c+'/ICOD/NO_APC/eigenvectors.npy',egvect_icod)
        
        np.save(path_c+'/ICOD/APC/eigenvalues.npy',egval_icod_apc)
        np.save(path_c+'/ICOD/APC/eigenvectors.npy',egvect_icod_apc)
        

def TaskMI(path_family,cutoff):
    if not(os.path.exists(path_family+cutoff+'/MI')):
        path_c = path_family+cutoff

        msacutoff = np.load(path_c+'/MSA_cutoff'+cutoff+'.npy')
        mimatrix = ComputeNJEInts(msacutoff, 0.001)
        egval_mi,egvect_mi = np.linalg.eigh(mimatrix) 
        mimatrix = APCNodiagMATLAB(mimatrix)
        egval_mi_apc,egvect_mi_apc = np.linalg.eigh(mimatrix)
        
        os.mkdir(path_c+'/MI')

        os.mkdir(path_c+'/MI/APC');os.mkdir(path_c+'/MI/NO_APC')

        np.save(path_c+'/MI/NO_APC/eigenvalues.npy',egval_mi)
        np.save(path_c+'/MI/NO_APC/eigenvectors.npy',egvect_mi)
        
        np.save(path_c+'/MI/APC/eigenvalues.npy',egval_mi_apc)
        np.save(path_c+'/MI/APC/eigenvectors.npy',egvect_mi_apc)
       
    
       
def TaskSCA(path_family,cutoff):
    if not(os.path.exists(path_family+cutoff+'/SCA')):
        path_c = path_family+cutoff

        msacutoff = np.load(path_c+'/MSA_cutoff'+cutoff+'.npy')
        

        msacutoff = msacutoff + 1
        scamatrix,_,_ = scaMat(msacutoff,seqw = 1,norm = 'frob')
        
        egval_sca,egvect_sca = np.linalg.eigh(scamatrix) 
        
        apc_scamatrix = APCNodiagMATLAB(scamatrix)
        egval_sca_apc,egvect_sca_apc = np.linalg.eigh(apc_scamatrix) 
        
        os.mkdir(path_c+'/SCA')

        os.mkdir(path_c+'/SCA/APC');os.mkdir(path_c+'/SCA/NO_APC')

        
        np.save(path_c+'/SCA/NO_APC/eigenvalues.npy',egval_sca)
        np.save(path_c+'/SCA/NO_APC/eigenvectors.npy',egvect_sca)
        
        np.save(path_c+'/SCA/APC/eigenvalues.npy',egval_sca_apc)
        np.save(path_c+'/SCA/APC/eigenvectors.npy',egvect_sca_apc)

def Conservation(msa):
    conservation =[]
    nbrseq, nbrsites = msa.shape
    nbrstates = np.unique(msa).shape[0]
    if nbrstates != 20:
        print('not 20 states')
    nstates = 20
    h_entropy = np.zeros(nbrsites)
    for j in range(0,nbrsites):
        val,cts = np.unique(msa[:,j],return_counts = True)

        h_entropy[j] = entropy(cts/nbrseq, base = nstates)
    

    conservation_singlesite = 1 - h_entropy
    
    return conservation_singlesite

def TaskConservation(path_family,cutoff):

    if not(os.path.exists(path_family+'/'+cutoff+'/Conservation')):
        os.mkdir(path_family+'/'+cutoff+'/Conservation')
        
        pathsave = path_family+'/'+cutoff+'/Conservation/'
        path_c = path_family+'/'+cutoff

        msacutoff = np.load(path_c+'/MSA_cutoff'+cutoff+'.npy')
        
        conservation = Conservation(msacutoff)

        np.save(pathsave+'conservation.npy',conservation)

def RemoveFilesDot(listdot):
    for f in listdot:
        if f.startswith('.'):
            listdot.remove(f)
    return listdot

        
def ComputeMethodCutoffMSAs(pathfolder,mthdstr, nparall):
    listfam = RemoveFilesDot(os.listdir(pathfolder))
    for folder in listfam:
        path_family = pathfolder+folder+'/FinalMSAs/cutoffs/'
        list_cutoffs = RemoveFilesDot(os.listdir(path_family))
        
        if mthdstr == 'conservation':
            Parallel(n_jobs=nparall)(delayed(TaskConservation)(path_family,cutoff) for cutoff in list_cutoffs)
        
        elif mthdstr == 'ICOD':
            Parallel(n_jobs=nparall)(delayed(TaskICOD)(path_family,cutoff) for cutoff in list_cutoffs)
        
        elif mthdstr == 'MI':
            Parallel(n_jobs=nparall)(delayed(TaskMI)(path_family,cutoff) for cutoff in list_cutoffs)
        
        elif mthdstr == 'SCA':
            Parallel(n_jobs=nparall)(delayed(TaskSCA)(path_family,cutoff) for cutoff in list_cutoffs)
        
        else:
            print('method name not corresponding to conservation, ICOD, MI or SCA')


###############################################################################    
##  SCA implementation from https://github.com/reynoldsk/pySCA
###############################################################################    
  

def alg2bin(alg, N_aa=20):
    ''' Translate an alignment of size M x L where the amino acids are represented 
    by numbers between 0 and N_aa (obtained using lett2num) to a sparse binary 
    array of size M x (N_aa x L). 
    
    :Example:
      >>> Abin = alg2bin(alg, N_aa=20) '''
    
    [N_seq, N_pos] = alg.shape
    Abin_tens = np.zeros((N_aa, N_pos, N_seq))
    for ia in range(N_aa):
        Abin_tens[ia,:,:] = (alg == ia+1).T
    Abin = sparsify(Abin_tens.reshape(N_aa*N_pos, N_seq, order='F').T)
    return Abin

def freq(alg, seqw=1, Naa=20, lbda=0, freq0=np.ones(20)/21):
    ''' 
    Compute amino acid frequencies for a given alignment.
    
    **Arguments:**
        -  `alg` = a MxL sequence alignment (converted using lett2num_) 
 
    .. _lett2num: scaTools.html#scaTools.lett2num
    **Keyword Arguments:**
        - `seqw` = a vector of sequence weights (1xM)
        - `Naa` = the number of amino acids
        - `lbda` = lambda parameter for setting the frequency of pseudo-counts (0 for no pseudo counts)
        - `freq0` = expected average frequency of amino acids at all positions
    **Returns:**
        -  `freq1` = the frequencies of amino acids at each position taken independently (Naa*L)
        -  `freq2` = the joint frequencies of amino acids at pairs of positions (freq2, Naa*L * Naa*L) 
        -  `freq0` = the average frequency of amino acids at all positions (Naa)
    :Example:
      >>> freq1, freq2, freq0 = freq(alg, seqw, lbda=lbda) 
   
    '''
    Nseq, Npos = alg.shape
    if type(seqw) == int and seqw == 1: seqw = np.ones((1,Nseq))
    seqwn = seqw/seqw.sum()
    al2d = alg2bin(alg, Naa)
    freq1 = seqwn.dot(np.array(al2d.todense()))[0]
    freq2 = np.array(al2d.T.dot(scipy.sparse.diags(seqwn[0], 0)).dot(al2d).todense())
    # Background:
    block = np.outer(freq0,freq0)
    freq2_bkg = np.zeros((Npos*Naa, Npos*Naa))
    for i in range(Npos): freq2_bkg[Naa*i:Naa*(i+1),Naa*i:Naa*(i+1)] = block
    # Regularizations:
    freq1_reg = (1-lbda)*freq1 + lbda*np.tile(freq0,Npos)
    freq2_reg = (1-lbda)*freq2 + lbda*freq2_bkg
    freq0_reg = freq1_reg.reshape(Npos, Naa).mean(axis=0)
    return freq1_reg, freq2_reg, freq0_reg

def posWeights(alg, seqw=1, lbda=0, N_aa = 20, freq0 = np.array([.073, .025, .050, .061, .042, .072,\
	.023, .053, .064, .089,.023, .043, .052, .040, .052, .073, .056, .063, .013, .033]), tolerance=1e-12):
    ''' Compute single-site measures of conservation, and the sca position weights, :math:`\\frac {\partial {D_i^a}}{\partial {f_i^a}}`
    **Arguments:**
         -  `alg` =  MSA, dimensions MxL, converted to numerical representation with lett2num_
         -  `seqw` = a vector of M sequence weights (default is uniform weighting)
         -  `lbda` = pseudo-counting frequencies, default is no pseudocounts
         -  `freq0` =  background amino acid frequencies :math:`q_i^a`
   
    **Returns:**
         -  `Wia` = positional weights from the derivation of a relative entropy, :math:`\\frac {\partial {D_i^a}}{\partial {f_i^a}}` (Lx20)
         -  `Dia` = the relative entropy per position and amino acid (Lx20)
         -   `Di` = the relative entropy per position (L) 
    :Example:
       >>> Wia, Dia, Di = posWeights(alg, seqw=1,freq0)
    
    '''
    N_seq, N_pos = alg.shape; 
    if type(seqw) == int and seqw == 1: seqw = np.ones((1,N_seq))
    freq1, freq2, _ = freq(alg, Naa=N_aa, seqw=seqw, lbda=lbda, freq0=freq0)
    # Overall fraction of gaps:
    theta = 1 - freq1.sum()/N_pos
    if theta<tolerance: theta = 0
    # Background frequencies with gaps:
    freqg0 = (1-theta)*freq0
    freq0v = np.tile(freq0,N_pos)
    iok = [i for i in range(N_pos*N_aa) if (freq1[i]>0 and freq1[i]<1)]
    # Derivatives of relative entropy per position and amino acid:
    Wia = np.zeros(N_pos*N_aa)
    Wia[iok] = abs(np.log((freq1[iok]*(1-freq0v[iok]))/((1-freq1[iok])*freq0v[iok])))
    # Relative entropies per position and amino acid:
    Dia = np.zeros(N_pos*N_aa)
    Dia[iok] = freq1[iok]*np.log(freq1[iok]/freq0v[iok])\
                   + (1-freq1[iok])*np.log((1-freq1[iok])/(1-freq0v[iok]))
    # Overall relative entropies per positions (taking gaps into account):
    Di = np.zeros(N_pos)
    for i in range(N_pos):
        freq1i = freq1[N_aa*i: N_aa*(i+1)]
        aok = [a for a in range(N_aa) if freq1i[a]>0]
        flogf = freq1i[aok]*np.log(freq1i[aok]/freqg0[aok])
        Di[i] = flogf.sum()
        freqgi = 1 - freq1i.sum()
        if freqgi > tolerance: Di[i] += freqgi*np.log(freqgi/theta)
    return Wia, Dia, Di

def scaMat(alg, seqw=1, norm='frob',lbda=0, freq0=np.ones(20)/21,):
    ''' Computes the SCA matrix.
       
     **Arguments:**
        - `alg` =  A MxL multiple sequence alignment, converted to numeric representation with lett2num_
    
     **Keyword Arguments:**
        -  `seqw` =  A vector of sequence weights (default: uniform weights)
        -  `norm` =   The type of matrix norm used for dimension reduction of the
                      SCA correlation tensor to a positional correlation matrix.
                      Use 'spec' for spectral norm and 'frob' for Frobenius
                      norm.  The frobenius norm is the default.
        -  `lbda` =  lambda parameter for setting the frequency of pseudo-counts (0 for no pseudo counts) 
        -  `freq0` = background expectation for amino acid frequencies
     **Returns:**
        -  `Cp` = the LxL SCA positional correlation matrix
        -  `tX` = the projected MxL alignment
        -  `projMat` = the projector
     :Example:
      >>> Csca, tX, projMat = scaMat(alg, seqw, norm='frob', lbda=0.03)
    '''
    N_seq, N_pos = alg.shape; N_aa = 20
    if type(seqw) == int and seqw == 1: seqw = np.ones((1,N_seq)) 
    freq1, freq2, freq0 = freq(alg, Naa=N_aa, seqw=seqw, lbda=lbda, freq0=freq0)
    W_pos = posWeights(alg, seqw, lbda)[0]
    tildeC = np.outer(W_pos,W_pos)*(freq2 - np.outer(freq1,freq1))
    # Positional correlations:
    Cspec = np.zeros((N_pos,N_pos))
    Cfrob = np.zeros((N_pos,N_pos))
    P = np.zeros((N_pos,N_pos,N_aa))
    for i in range(N_pos):
        for j in range(i,N_pos):
            u,s,vt = np.linalg.svd(tildeC[N_aa*i:N_aa*(i+1), N_aa*j:N_aa*(j+1)])
            Cspec[i,j] = s[0]
            Cfrob[i,j] = np.sqrt(sum(s**2))
            P[i,j,:] = np.sign(np.mean(u[:,0]))*u[:,0]
            P[j,i,:] = np.sign(np.mean(u[:,0]))*vt[0,:].T
    Cspec += np.triu(Cspec,1).T
    Cfrob += np.triu(Cfrob,1).T
    # Projector:
    al2d = np.array(alg2bin(alg).todense())
    tX = np.zeros((N_seq,N_pos))
    Proj = W_pos*freq1
    ProjMat = np.zeros((N_pos,N_aa))
    for i in range(N_pos):
        Projati = Proj[N_aa*i:N_aa*(i+1)] 
        if sum(Projati**2) > 0:
            Projati /= np.sqrt(sum(Projati**2))
        ProjMat[i,:] = Projati
        tX[:,i] = al2d[:,N_aa*i:N_aa*(i+1)].dot(Projati.T)
    if norm == 'frob' : Cspec = Cfrob
    return Cspec, tX, Proj

    
if __name__ == '__main__':
                
    if mp.get_start_method() == "spawn":
        import sys
        sys.exit(0)
    
    
    backend = 'multiprocessing'
    path_folder ='./MSAs/'
    listfam = os.listdir(path_folder)
    refseqDMS_id = 'Reference_sequence'
    
    # UN-/COMMENT WHAT IS NEEDED
    
    ###############################################################################
    ## GENERATE MSAs AT DIFFERENT CUTOFFS FROM JACKHMMER MSA
    ###############################################################################
    # for folder in listfam:
    #     if not(folder.startswith('.')):
    #         MSA_Cutoffs(path_folder,folder,refseqDMS_id)
    
        
    
    ###############################################################################
    ## COMPUTE ICOD, MI AND SCA ON MSAs AT DIFFERENT CUTOFFS
    ###############################################################################
    
    #choose appropriate method
    
    mthdstr = 'conservation'
    # mthdstr = 'ICOD'
    # mthdstr = 'MI'
    # mthdstr = 'SCA'
    
    #number of cpu to parallelise
    nparall = 4
    
    ComputeMethodCutoffMSAs(path_folder,mthdstr, nparall)
    

