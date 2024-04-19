#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze sectors
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from tqdm import tqdm
from numba import jit
import matplotlib.text as mpl_text
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
class TextHandlerB(HandlerBase):
    def create_artists(self, legend, text ,xdescent, ydescent,
                        width, height, fontsize, trans):
        tx = mpl_text.Text(0,height/2, text, fontsize=fontsize,
                  ha="left", va="center", fontweight="bold")
        return [tx]

from scipy.stats import entropy
from matplotlib.widgets import Button
import sklearn
import sklearn.metrics

import matplotlib.gridspec as gridspec
import scipy.stats
from scipy.sparse import csr_matrix as sparsify
from matplotlib.patches import Rectangle
import matplotlib as mpl

Legend.update_default_handler_map({str : TextHandlerB()})
colorspal6 = [(64, 83, 211), (221, 179, 16), (181, 29, 20), (0, 190, 255), (251, 73, 176), (0, 178, 93), (202, 202, 202)]
colorspal12 = [(235, 172, 35), (184, 0, 88), (0, 140, 249), (0, 110, 0), (0, 187, 173), (209, 99, 230), (178, 69, 2), (255, 146, 135), (89, 84, 214), (0, 198, 248), (135, 133, 0), (0, 167, 108), (189, 189, 189)]
newcolorspal =[]
newcolorspal12 =[]
newcolorspal_shades = []
for t in colorspal6:
    newcolorspal.append(tuple(ti/255 for ti in t))

for t in colorspal12:
    newcolorspal12.append(tuple(ti/255 for ti in t))

def Addtint(currentR,tint_factor):
    return currentR + (255 - currentR) * tint_factor

for r in colorspal6:
    
    newcolorspal_shades.append(tuple(Addtint(ti,0.5)/255 for ti in r))
    
fontsize_test = 23

plt.rcParams.update({'font.size': fontsize_test})

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 10
plt.rcParams['xtick.minor.width'] = 2

plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 10
plt.rcParams['ytick.minor.width'] = 2
plt.rc('xtick',labelsize=fontsize_test)
plt.rc('ytick',labelsize=fontsize_test)


def OpenFile(path,parametersname1,val1):
    
    file=h5py.File(path,'r')
    sequences = np.array(file['Chains'])
    parameters = np.array(file[parametersname1])

    val = np.array(file[val1])
    delta =np.array(file['Delta'])
    file.close()
    
    return sequences,delta,parameters,val

def OpenFile2(path,parametersname1,val1,val2):
    
    file=h5py.File(path,'r')
    sequences = np.array(file['Chains'])
    parameters = np.array(file[parametersname1])

    val1 = np.array(file[val1])
    val2 = np.array(file[val2])
    delta =np.array(file['Delta'])
    file.close()
    
    return sequences,delta,parameters,val1,val2

###############################################################################    
##  SCA implementation from https://github.com/reynoldsk/pySCA
###############################################################################    
  
def alg2bin(alg, N_aa=2):
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
# @jit(nopython=True)
def freq(alg, seqw=1, Naa=2, lbda=0, freq0=np.ones(2)/2):
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

# @jit(nopython=True)
def posWeights(alg, seqw=1, lbda=0, N_aa = 2, freq0 = np.array([0.5,0.5]), tolerance=1e-12):
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

# @jit(nopython=True)
def scaMat(alg, seqw=1, norm='frob',lbda=0, freq0=np.ones(2)/2,):
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
    N_seq, N_pos = alg.shape; N_aa = 2
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

    
def RecoveryOneVector(vector,delta):
    
    return np.sum(abs(delta*vector)) /(np.sqrt(np.dot(vector,vector)) * np.sqrt(np.dot(delta,delta)))


def ComputeCorrelationMatrix2(mat, pseudocount):
    
    nbr_spins = len(mat[0,:])
    nbr_chains = len(mat[:,0])
    mat = np.array(mat,ndmin = 2, dtype = np.float64)
    average_spin = np.average(mat, axis = 0)[:,None]
    
    directcorr = np.dot(mat.T, mat)
    directcorr *= np.true_divide(1, nbr_chains, dtype = np.float64)
    correlation_matrix = np.dot(1.0-pseudocount, directcorr) - np.dot(pow(1-pseudocount,2),np.outer(average_spin.T, average_spin)) + np.dot(pseudocount,np.identity(nbr_spins))
    
    return correlation_matrix

def Eigenvalues(filepath,parameterkey,paramvalue,paramkey2,pseudocount):
    file = h5py.File(filepath,'r')
    sequences = np.array(file['Chains'])
    parameters = np.array(file[parameterkey])
    paraval2 = np.array(file[paramkey2])
    file.close()
    chains = sequences[parameters==paramvalue,:,:][0,:,:]
    paraval1 = parameters[parameters ==paramvalue]
    covariance_matrix = ComputeCorrelationMatrix2(chains, pseudocount)
    covariance_matrix_0pseudocount = ComputeCorrelationMatrix2(chains, 0)
    
    eigenvalues_cov, eigenvectors_cov = np.linalg.eigh(covariance_matrix_0pseudocount)

    icod = np.linalg.inv(covariance_matrix)
    np.fill_diagonal(icod,0)
    eigenvalues, eigenvectors = np.linalg.eigh(icod)
    
    return np.flip(eigenvalues_cov),np.flip(eigenvalues),paraval1,paraval2

def EigenvaluesEigenvectorsSCA(filepath,parameterkey,paramvalue,paramkey2,pseudocount):
    file = h5py.File(filepath,'r')
    sequences = np.array(file['Chains'])
    parameters = np.array(file[parameterkey])
    paraval2 = np.array(file[paramkey2])
    file.close()
    chains = sequences[parameters==paramvalue,:,:][0,:,:]
    paraval1 = parameters[parameters ==paramvalue]
    chainstransf = (chains + 1)/2
    scamatrix,_,_ = scaMat(chainstransf,seqw = 1,norm = 'frob')
    eigenvalues_cov, eigenvectors_cov = np.linalg.eigh(scamatrix)

    return np.flip(eigenvalues_cov),np.flip(eigenvectors_cov,axis =1),paraval1,paraval2


def Eigenvectors(filepath,parameterkey,paramvalue,paramkey2,pseudocount):
    file = h5py.File(filepath,'r')
    sequences = np.array(file['Chains'])
    parameters = np.array(file[parameterkey])
    paraval2 = np.array(file[paramkey2])
    file.close()
    chains = sequences[parameters==paramvalue,:,:][0,:,:]
    paraval1 = parameters[parameters ==paramvalue]
    covariance_matrix = ComputeCorrelationMatrix2(chains, pseudocount)
    covariance_matrix_0pseudocount = ComputeCorrelationMatrix2(chains, 0)
    
    eigenvalues_cov, eigenvectors_cov = np.linalg.eigh(covariance_matrix_0pseudocount)

    icod = np.linalg.inv(covariance_matrix)
    np.fill_diagonal(icod,0)
    eigenvalues, eigenvectors = np.linalg.eigh(icod)
    
    return np.flip(eigenvectors_cov,axis=1 ) ,np.flip(eigenvectors,axis =1),paraval1,paraval2

def EigenvectorsSavePhylogeny(filepath,pseudocount):
    

    file = h5py.File(filepath,'r')
    sequences = np.array(file['Chains'])
    allchains = np.array(file['allchains'])

    number_generations =file['NumberGenerations'][()]

    file.close()
    chains = sequences

    covariance_matrix = ComputeCorrelationMatrix2(chains, pseudocount)
    covariance_matrix_0pseudocount = ComputeCorrelationMatrix2(chains, 0)
    
    eigenvalues_cov, eigenvectors_cov = np.linalg.eigh(covariance_matrix_0pseudocount)

    icod = np.linalg.inv(covariance_matrix)
    np.fill_diagonal(icod,0)
    eigenvalues, eigenvectors = np.linalg.eigh(icod)
    
    chainstransf = (chains + 1)/2
    scamatrix,_,_ = scaMat(chainstransf,seqw = 1,norm = 'frob')
    eigenvalues_sca, eigenvectors_sca = np.linalg.eigh(scamatrix)
    
    return np.flip(eigenvectors_cov,axis=1),np.flip(eigenvectors,axis =1),np.flip(eigenvectors_sca,axis =1),allchains,number_generations
    
def RecoveryVSparam(chainsparam, parameters,delta, pseudocount):
    
    nbr_params= parameters.shape[0]
    
    recovery_cov_lambdamin = np.zeros(nbr_params);recovery_cov_lambdamax = np.zeros(nbr_params)
    recovery_icod_Lambdamin = np.zeros(nbr_params);recovery_icod_Lambdamax = np.zeros(nbr_params)
    
    for idxp,param in tqdm(enumerate(parameters)):

        chains = chainsparam[idxp,:,:].copy()
        covariance_matrix = ComputeCorrelationMatrix2(chains, pseudocount)
        covariance_matrix0pseudocount = ComputeCorrelationMatrix2(chains, 0)
    
        eigenvalues_cov, eigenvectors_cov = np.linalg.eigh(covariance_matrix0pseudocount)

        recovery_cov_lambdamin[idxp] = RecoveryOneVector(eigenvectors_cov[:,0],delta)
        recovery_cov_lambdamax[idxp] = RecoveryOneVector(eigenvectors_cov[:,-1],delta)
        
        icod = np.linalg.inv(covariance_matrix)
        np.fill_diagonal(icod,0)
        eigenvalues, eigenvectors = np.linalg.eigh(icod)
        recovery_icod_Lambdamax[idxp] = RecoveryOneVector(eigenvectors[:,-1], delta)
        recovery_icod_Lambdamin[idxp] = RecoveryOneVector(eigenvectors[:,0], delta)
    
    return recovery_cov_lambdamax, recovery_cov_lambdamin, recovery_icod_Lambdamax, recovery_icod_Lambdamin


def RecoveryVSparamSCA(chainsparam, parameters,delta, pseudocount):
    
    nbr_params= parameters.shape[0]
    
    recovery_cov_lambdamin = np.zeros(nbr_params);recovery_cov_lambdamax = np.zeros(nbr_params)
 
    for idxp,param in tqdm(enumerate(parameters)):

        chains = chainsparam[idxp,:,:].copy()
        
        chains_transf = (chains+1)/2
        
        scamatrix,_,_ = scaMat(chains_transf,seqw = 1,norm = 'frob')
        
    
        eigenvalues_cov, eigenvectors_cov = np.linalg.eigh(scamatrix)

        recovery_cov_lambdamin[idxp] = RecoveryOneVector(eigenvectors_cov[:,0],delta)
        recovery_cov_lambdamax[idxp] = RecoveryOneVector(eigenvectors_cov[:,-1],delta)
        
    return recovery_cov_lambdamax, recovery_cov_lambdamin

def Conservation(msa):
    conservation =[]
    nbrseq, nbrsites = msa.shape
    nbrstates = np.unique(msa).shape[0]
    if nbrstates != 2:
        print('not 2 states')

    nstates = 2

    h_entropy = np.zeros(nbrsites)
    for j in range(0,nbrsites):
        val,cts = np.unique(msa[:,j],return_counts = True)
        h_entropy[j] = entropy(cts/nbrseq, base = nstates)
    

    conservation_singlesite = 1 - h_entropy
    
    return conservation_singlesite


def RecoveryAUCVSparamConservation(chainsparam,parameters,delta):
    
    nbr_params= parameters.shape[0]
    
    recovery_cons = np.zeros(nbr_params)
    AUC_cons = np.zeros(nbr_params)
    boolean_delta = np.zeros(delta.shape)
    boolean_delta[:20] = 1
    
    for idxp,param in tqdm(enumerate(parameters)):

        chains = chainsparam[idxp,:,:].copy()

        conservation = Conservation(chains)

        recovery_cons[idxp] = RecoveryOneVector(delta,conservation)

        AUC_cons[idxp] = AUC(boolean_delta,conservation)

    return recovery_cons, AUC_cons


def AUC(gt,sector):
    
    fpr,tpr,_ = sklearn.metrics.roc_curve(gt, sector)
    return 2*np.abs(sklearn.metrics.auc(fpr, tpr)-0.5)

def AUCVSparam(chainsparam,parameters,delta, pseudocount):
    # pdb.set_trace()
    nbr_params = parameters.shape[0]
    
    auc_cov_lambdamin = np.zeros(nbr_params);auc_cov_lambdamax = np.zeros(nbr_params)
    auc_icod_Lambdamin = np.zeros(nbr_params);auc_icod_Lambdamax = np.zeros(nbr_params)
    
    
    boolean_delta = np.zeros(delta.shape)
    boolean_delta[:20] = 1
    
    for idxp,param in tqdm(enumerate(parameters)):

        chains = chainsparam[idxp,:,:].copy()
        covariance_matrix = ComputeCorrelationMatrix2(chains, pseudocount)
        covariance_matrix0pseudocount = ComputeCorrelationMatrix2(chains, 0)
    
        eigenvalues_cov, eigenvectors_cov = np.linalg.eigh(covariance_matrix0pseudocount)
        
        pcorr,_ = scipy.stats.pearsonr(eigenvectors_cov[:,0], delta)
        auc_cov_lambdamin[idxp] = AUC(boolean_delta, eigenvectors_cov[:,0]*np.sign(pcorr))
        
        pcorr,_ = scipy.stats.pearsonr(eigenvectors_cov[:,-1], delta)
        auc_cov_lambdamax[idxp] = AUC(boolean_delta, eigenvectors_cov[:,-1]*np.sign(pcorr))
        
        
        icod = np.linalg.inv(covariance_matrix)
        np.fill_diagonal(icod,0)
        eigenvalues, eigenvectors = np.linalg.eigh(icod)
        pcorr,_ = scipy.stats.pearsonr(eigenvectors[:,-1], delta)
        auc_icod_Lambdamax[idxp] = AUC(boolean_delta, eigenvectors[:,-1]*np.sign(pcorr))
        pcorr,_ = scipy.stats.pearsonr(eigenvectors[:,0], delta)
        auc_icod_Lambdamin[idxp] = AUC(boolean_delta, eigenvectors[:,0]*np.sign(pcorr))
    
    return auc_cov_lambdamax, auc_cov_lambdamin, auc_icod_Lambdamax, auc_icod_Lambdamin

def AUCVSparamSCA(chainsparam,parameters,delta, pseudocount):

    nbr_params = parameters.shape[0]
    auc_cov_lambdamin = np.zeros(nbr_params);auc_cov_lambdamax = np.zeros(nbr_params)
    
    boolean_delta = np.zeros(delta.shape)
    boolean_delta[:20] = 1
    
    for idxp,param in tqdm(enumerate(parameters)):

        chains = chainsparam[idxp,:,:].copy()
        chains_transf = (chains+1)/2
        
        scamatrix,_,_ = scaMat(chains_transf,seqw = 1,norm = 'frob')
        
        eigenvalues_cov, eigenvectors_cov = np.linalg.eigh(scamatrix)
        
        pcorr,_ = scipy.stats.pearsonr(eigenvectors_cov[:,0], delta)
        auc_cov_lambdamin[idxp] = AUC(boolean_delta, eigenvectors_cov[:,0]*np.sign(pcorr))
        
        pcorr,_ = scipy.stats.pearsonr(eigenvectors_cov[:,-1], delta)
        auc_cov_lambdamax[idxp] = AUC(boolean_delta, eigenvectors_cov[:,-1]*np.sign(pcorr))
        
    return auc_cov_lambdamax, auc_cov_lambdamin


def Averaging(pathfolder,parametername,valname,pseudocount):
    
    listfiles = os.listdir(pathfolder)

    list_C_lmax = [];list_C_lmin = []
    list_icod_Lmax = [];list_icod_Lmin = []
    
    for f in listfiles:
        if f.startswith('.'):
            listfiles.remove(f)
    for f in listfiles:
        sequences,delta,parameters,val = OpenFile(pathfolder+f, parametername, valname)

        cov_lmax,cov_lmin, icodLmax,icodLmin = RecoveryVSparam(sequences, parameters, delta, pseudocount)
        
        list_C_lmax.append(cov_lmax)
        list_C_lmin.append(cov_lmin)
        
        list_icod_Lmax.append(icodLmax)
        list_icod_Lmin.append(icodLmin)
        
    return Av(list_C_lmax),Av(list_C_lmin),Av(list_icod_Lmax),Av(list_icod_Lmin),parameters,val

def AveragingSCA(pathfolder,parametername,valname,pseudocount):
    
    listfiles = os.listdir(pathfolder)
    # pseudocount = 0.1
    
    list_C_lmax = [];list_C_lmin = []

    
    for f in listfiles:
        if f.startswith('.'):
            listfiles.remove(f)
    for f in listfiles:
        sequences,delta,parameters,val = OpenFile(pathfolder+f, parametername, valname)
        # sequences = Invertseq(sequences)
        cov_lmax,cov_lmin = RecoveryVSparamSCA(sequences, parameters, delta, pseudocount)
        
        list_C_lmax.append(cov_lmax)
        list_C_lmin.append(cov_lmin)

        
    return Av(list_C_lmax),Av(list_C_lmin),parameters,val

def AveragingAUC(pathfolder,parametername,valname,pseudocount):
    
    listfiles = os.listdir(pathfolder)
    # pseudocount = 0.1
    
    list_C_lmax = [];list_C_lmin = []
    list_icod_Lmax = [];list_icod_Lmin = []
    
    for f in listfiles:
        if f.startswith('.'):
            listfiles.remove(f)
    for f in listfiles:
        sequences,delta,parameters,val = OpenFile(pathfolder+f, parametername, valname)
        # sequences = Invertseq(sequences)
        cov_lmax,cov_lmin, icodLmax,icodLmin = AUCVSparam(sequences, parameters,delta, pseudocount)
        
        list_C_lmax.append(cov_lmax)
        list_C_lmin.append(cov_lmin)
        
        list_icod_Lmax.append(icodLmax)
        list_icod_Lmin.append(icodLmin)
        
    return Av(list_C_lmax),Av(list_C_lmin),Av(list_icod_Lmax),Av(list_icod_Lmin),parameters,val

def AveragingAUCSCA(pathfolder,parametername,valname,pseudocount):
    
    listfiles = os.listdir(pathfolder)
    # pseudocount = 0.1
    
    list_C_lmax = [];list_C_lmin = []

    
    for f in listfiles:
        if f.startswith('.'):
            listfiles.remove(f)
    for f in listfiles:
        sequences,delta,parameters,val = OpenFile(pathfolder+f, parametername, valname)
        # sequences = Invertseq(sequences)
        cov_lmax,cov_lmin = AUCVSparamSCA(sequences, parameters,delta, pseudocount)
        
        list_C_lmax.append(cov_lmax)
        list_C_lmin.append(cov_lmin) 
    return Av(list_C_lmax),Av(list_C_lmin)


def Averaging2(pathfolder,parametername,valname1,valname2,pseudocount):
    
    listfiles = os.listdir(pathfolder)
    # pseudocount = 0.1
    
    list_C_lmax = [];list_C_lmin = []
    list_icod_Lmax = [];list_icod_Lmin = []
    
    for f in listfiles:
        if f.startswith('.'):
            listfiles.remove(f)
    for f in listfiles:
        sequences,delta,parameters,val1,val2 = OpenFile2(pathfolder+f, parametername, valname1,valname2)
        
        cov_lmax,cov_lmin, icodLmax,icodLmin = RecoveryVSparam(sequences, parameters, delta, pseudocount)
        
        list_C_lmax.append(cov_lmax)
        list_C_lmin.append(cov_lmin)
        
        list_icod_Lmax.append(icodLmax)
        list_icod_Lmin.append(icodLmin)
        
    return Av(list_C_lmax),Av(list_C_lmin),Av(list_icod_Lmax),Av(list_icod_Lmin),parameters,val1,val2

def Averaging2SCA(pathfolder,parametername,valname1,valname2,pseudocount):
    
    listfiles = os.listdir(pathfolder)
    # pseudocount = 0.1
    
    list_C_lmax = [];list_C_lmin = []

    
    for f in listfiles:
        if f.startswith('.'):
            listfiles.remove(f)
    for f in listfiles:
        sequences,delta,parameters,val1,val2 = OpenFile2(pathfolder+f, parametername, valname1,valname2)
        
        cov_lmax,cov_lmin = RecoveryVSparamSCA(sequences, parameters, delta, pseudocount)
        
        list_C_lmax.append(cov_lmax)
        list_C_lmin.append(cov_lmin)

    return Av(list_C_lmax),Av(list_C_lmin),parameters,val1,val2

def Averaging2AUC(pathfolder,parametername,valname1,valname2,pseudocount):
    
    listfiles = os.listdir(pathfolder)
    # pseudocount = 0.1
    
    list_C_lmax = [];list_C_lmin = []
    list_icod_Lmax = [];list_icod_Lmin = []
    
    for f in listfiles:
        if f.startswith('.'):
            listfiles.remove(f)
    for f in listfiles:
        sequences,delta,parameters,val1,val2 = OpenFile2(pathfolder+f, parametername, valname1,valname2)
        
        cov_lmax,cov_lmin, icodLmax,icodLmin = AUCVSparam(sequences, parameters,delta, pseudocount)
        
        list_C_lmax.append(cov_lmax)
        list_C_lmin.append(cov_lmin)
        
        list_icod_Lmax.append(icodLmax)
        list_icod_Lmin.append(icodLmin)
        
    return Av(list_C_lmax),Av(list_C_lmin),Av(list_icod_Lmax),Av(list_icod_Lmin),parameters,val1,val2

def Averaging2AUCSCA(pathfolder,parametername,valname1,valname2,pseudocount):
    listfiles = os.listdir(pathfolder)
    # pseudocount = 0.1
    list_C_lmax = [];list_C_lmin = []
    for f in listfiles:
        if f.startswith('.'):
            listfiles.remove(f)
    for f in listfiles:
        sequences,delta,parameters,val1,val2 = OpenFile2(pathfolder+f, parametername, valname1,valname2)
        cov_lmax,cov_lmin = AUCVSparamSCA(sequences, parameters,delta, pseudocount)
    
        list_C_lmax.append(cov_lmax)
        list_C_lmin.append(cov_lmin)
        
    return Av(list_C_lmax),Av(list_C_lmin),parameters,val1,val2

def Averaging2Conservation(pathfolder,parametername,valname1,valname2):
    listfiles = os.listdir(pathfolder)
    # pseudocount = 0.1
    list_cons_rec = [];list_cons_auc = []
    for f in listfiles:
        if f.startswith('.'):
            listfiles.remove(f)
    for f in listfiles:
        sequences,delta,parameters,val1,val2 = OpenFile2(pathfolder+f, parametername, valname1,valname2)
        recovery_cons, AUC_cons = RecoveryAUCVSparamConservation(sequences,parameters,delta)
        list_cons_rec.append(recovery_cons)
        list_cons_auc.append(AUC_cons)
        
    return Av(list_cons_rec),Av(list_cons_auc),parameters,val1,val2

def AveragingConservation(pathfolder,parametername,valname):
    listfiles = os.listdir(pathfolder)
    list_cons_rec = [];list_cons_auc = []

    for f in listfiles:
        if f.startswith('.'):
            listfiles.remove(f)
    for f in listfiles:
        sequences,delta,parameters,val = OpenFile(pathfolder+f, parametername, valname)
        # sequences = Invertseq(sequences)
        recovery_cons, AUC_cons = RecoveryAUCVSparamConservation(sequences,parameters,delta)
        
        list_cons_rec.append(recovery_cons)
        list_cons_auc.append(AUC_cons)
   
    return Av(list_cons_rec),Av(list_cons_auc),parameters,val

def Av(listrec):
    
    arr = np.array(listrec)
    return np.average(arr,axis = 0)


def LoadRec(path):
    C_lmax = np.load(path+'C_lmax.npy')
    C_lmin = np.load(path+'C_lmin.npy')
    icod_Lmax = np.load(path+'ICOD_Lmax.npy')
    icod_Lmin = np.load(path+'ICOD_Lmin.npy')
    sca_lmax = np.load(path+'SCA_lmax.npy')
    sca_lmin = np.load(path+'SCA_lmin.npy')
    conservation = np.load(path+'conservation.npy')
    return C_lmax,C_lmin,icod_Lmax,icod_Lmin,sca_lmax,sca_lmin,conservation

def LoadFiles(path):
    
    filelist= os.listdir(path)
    for f in filelist:
        if f.startswith('.'):
            filelist.remove(f)
    return filelist
    
def RandomExpectation(path):
    filelist = LoadFiles(path)

    file = h5py.File(path+filelist[0],'r')
    delta = np.array(file['Delta'])
    file.close()
    randomexpectation = np.sqrt(2/(np.pi*delta.shape[0]))*np.sum(np.abs(delta))/(np.sqrt(np.dot(delta,delta)))
    return randomexpectation   

def PlotRecvsM():
    pthfolder = './sequences/phylogeny_K40_alpha90_M1_50_100real/'
    ptheq = './sequences/EQ_K40_alpham300_300_100real/'
    parametername='Mutations'
    valname1 = 'Alpha'
    valname2 = 'Kappa'
    pseudocount = 0.00001
    
    pth1real_eq = ptheq+'2023_04_24_11_47_24_Sectors_nspins200_flips3000_nseq2048_seed_34_deltanormal_Alphamin-300_Alphamax300_K40_realnbr0.h5'
   
    _,_,parameq,_ = OpenFile(pth1real_eq,'Alpha', 'Kappa')
    patheqmain = './results/recovery/EQ_K40_alpham300_300_100real/'
    C_lmaxeq,C_lmineq,icod_Lmaxeq,icod_Lmineq,_,_,conservation_eq = LoadRec(patheqmain)

    pathmain = './results/recovery/phylogeny_K40_alpha90_M1_50_100real/'

    C_lmax,C_lmin,icod_Lmax,icod_Lmin,_,_,conservation = LoadRec(pathmain)

    parameters = np.arange(50)+1
    
    random_expectation = RandomExpectation(pthfolder)
    
    val1 = 90
    val2 = 40
    lwdth = 2
    idx_cl1 = 3
    idx_cl2 = 5
    idx_cl3 = 6
    fig = plt.figure(figsize = (14.22,  7.16))

    plt.plot(parameters,C_lmin,'--^', color = newcolorspal[idx_cl1],linewidth = lwdth,label = 'C $\lambda_{min}$')
    plt.plot(parameters,C_lmax,'--^',color = newcolorspal12[idx_cl2],linewidth = lwdth,label = 'C $\lambda_{max}$')

    plt.plot(parameters,random_expectation*np.ones(parameters.shape),':',color = 'black',linewidth = lwdth,label = 'Random expectation',zorder = -1)
    
    
    plt.hlines(C_lmaxeq[parameq ==val1],parameters[0],parameters[-1],linestyle = '--',color = newcolorspal12[idx_cl2],alpha = 0.6,linewidth = lwdth)
    plt.hlines(icod_Lmaxeq[parameq ==val1],parameters[0],parameters[-1],linestyle = '-',color = newcolorspal[idx_cl1], alpha = 0.6,linewidth = lwdth)
    plt.hlines(icod_Lmineq[parameq ==val1],parameters[0],parameters[-1],linestyle ='-',color = newcolorspal12[idx_cl2], alpha = 0.6,linewidth =lwdth)
    plt.hlines(conservation_eq[parameq ==val1],parameters[0],parameters[-1],linestyle ='--',color = newcolorspal12[idx_cl3], alpha = 0.6,linewidth =lwdth)
    
    
    plt.plot(parameters,conservation,'--d',color = newcolorspal12[idx_cl3],linewidth = lwdth,label = 'ICOD $\Lambda_{min}$',zorder = -1)
    
    plt.plot(parameters,icod_Lmax,'-o', color = newcolorspal[idx_cl1],linewidth = lwdth,label = 'ICOD $\Lambda_{max}$')
    plt.plot(parameters,icod_Lmin,'-o',color = newcolorspal12[idx_cl2],linewidth = lwdth,label = 'ICOD $\Lambda_{min}$')
    plt.ylabel('Recovery')
    plt.xlabel(r'Number of accepted mutations $\mu$')

    plt.xlim([0,51])
    custom_lines = ["Phylogeny: ",
                    Line2D([0], [0], color=newcolorspal[idx_cl1], lw=lwdth,linestyle = '-',marker = 'o'),
                    Line2D([0], [0], color=newcolorspal[idx_cl1], lw=lwdth,linestyle = '--',marker = '^'),
                    Line2D([0], [0], color=newcolorspal12[idx_cl2], lw=lwdth,linestyle = '-',marker = 'o'),
                    Line2D([0], [0], color=newcolorspal12[idx_cl2], lw=lwdth,linestyle = '--',marker = '^'),
                    Line2D([0], [0], color=newcolorspal12[idx_cl3], lw=lwdth,linestyle = '--',marker = 'd'),
                    " ",
                    "No phylogeny: ",
                    Line2D([0], [0], color=newcolorspal[idx_cl1], lw=lwdth,alpha = 0.6,linestyle = '-'),
                    Line2D([0], [0], color=newcolorspal12[idx_cl2], lw=lwdth,alpha = 0.6,linestyle = '-'),
                    Line2D([0], [0], color=newcolorspal12[idx_cl2], lw=lwdth,alpha = 0.6,linestyle = '--'),
                    Line2D([0], [0], color=newcolorspal12[idx_cl3], lw=lwdth,alpha = 0.6,linestyle = '--'),
                    " ",
                    Line2D([0], [0], color='black', lw=lwdth,linestyle = ':')]
    
    labels = ['','ICOD $\Lambda_{max}$','C $\lambda_{min}$','ICOD $\Lambda_{min}$','C $\lambda_{max}$','Conservation','','','ICOD $\Lambda_{max}$, C $\lambda_{min}$','ICOD $\Lambda_{min}$','C $\lambda_{max}$','Conservation','','Null model']#,'$\mu = 5$ + APC','$\mu = 15$ + APC','eq + APC' ]
    
    plt.legend(custom_lines,labels,bbox_to_anchor = (1, 1.05), loc='upper left')

    fig.subplots_adjust(wspace=0.2,hspace = 0.5,top= 0.975, bottom = 0.145, left = 0.08,right = 0.685)

def PlotRecvsM_SCA():
    pthfolder = './sequences/phylogeny_K40_alpha90_M1_50_100real/'
    ptheq = './sequences/EQ_K40_alpham300_300_100real/'
    parametername='Mutations'
    valname1 = 'Alpha'
    valname2 = 'Kappa'
    pseudocount = 0.00001
    # C_lmax,C_lmin, icod_Lmax,icod_Lmin,parameters,val1,val2 = Averaging2(pthfolder, parametername, valname1,valname2, pseudocount)
    pth1real_eq = ptheq+'2023_04_24_11_47_24_Sectors_nspins200_flips3000_nseq2048_seed_34_deltanormal_Alphamin-300_Alphamax300_K40_realnbr0.h5'
    _,_,parameq,_ = OpenFile(pth1real_eq,'Alpha', 'Kappa')
    patheqmain = './results/recovery/EQ_K40_alpham300_300_100real/'
    C_lmaxeq,C_lmineq,icod_Lmaxeq,icod_Lmineq,sca_lmaxeq,sca_lmineq,conseq = LoadRec(patheqmain)
    # C_lmaxeq,C_lmineq,icod_Lmaxeq,icod_Lmineq,parameq,valeq = Averaging(ptheq, 'Alpha', 'Kappa', pseudocount)
    
    pathmain = './results/recovery/phylogeny_K40_alpha90_M1_50_100real/'
    # np.save(pathmain+'C_lmax.npy',C_lmax)
    # np.save(pathmain+'C_lmin.npy',C_lmin)
    # np.save(pathmain+'ICOD_Lmax.npy',icod_Lmax)
    # np.save(pathmain+'ICOD_Lmin.npy',icod_Lmin)
    C_lmax,C_lmin,icod_Lmax,icod_Lmin,sca_lmax,sca_lmin,cons = LoadRec(pathmain)

    parameters = np.arange(50)+1
    
    random_expectation = RandomExpectation(pthfolder)
    
    val1 = 90
    val2 = 40
    lwdth = 2
    idx_cl1 = 3
    idx_cl2 = 5
    idx_cl3 = 6
    fig = plt.figure(figsize = (14.22,  7.16))

    plt.plot(parameters,random_expectation*np.ones(parameters.shape),':',color = 'black',linewidth = lwdth,label = 'Random expectation',zorder = -1)

    plt.plot(parameters,icod_Lmax,'-', color = newcolorspal[idx_cl1],linewidth = lwdth,label = 'ICOD $\Lambda_{max}$')
    plt.plot(parameters,icod_Lmin,'-',color = newcolorspal12[idx_cl2],linewidth = lwdth,label = 'ICOD $\Lambda_{min}$')
    
    plt.plot(parameters,sca_lmax,'-s', color = newcolorspal[idx_cl1],linewidth = lwdth,label = 'ICOD $\Lambda_{max}$')
    plt.plot(parameters,sca_lmin,'-s',color = newcolorspal12[idx_cl2],linewidth = lwdth,label = 'ICOD $\Lambda_{min}$')
    
    plt.plot(parameters,cons,'--d',color = newcolorspal12[idx_cl3],linewidth = lwdth,label = 'ICOD $\Lambda_{min}$')
    
    
    plt.ylabel('Recovery')
    plt.xlabel(r'Number of accepted mutations $\mu$')

    plt.xlim([0,51])

    custom_lines = [Line2D([0], [0], color=newcolorspal[idx_cl1], lw=lwdth,linestyle = '-'),
                    Line2D([0], [0], color=newcolorspal[idx_cl1], lw=lwdth,linestyle = '-',marker = 's'),
                    Line2D([0], [0], color=newcolorspal12[idx_cl2], lw=lwdth,linestyle = '-'),
                    Line2D([0], [0], color=newcolorspal12[idx_cl2], lw=lwdth,linestyle = '-',marker = 's'),
                    Line2D([0], [0], color=newcolorspal12[idx_cl3], lw=lwdth,linestyle = '--',marker = 'd'),
                    
                    " ",
                    Line2D([0], [0], color='black', lw=lwdth,linestyle = ':')]
    
    labels = ['ICOD $\Lambda_{max}$','SCA $\lambda_{max}$','ICOD $\Lambda_{min}$','SCA $\lambda_{min}$','Conservation','','Null model']#,'$\mu = 5$ + APC','$\mu = 15$ + APC','eq + APC' ]
    
    plt.legend(custom_lines,labels,bbox_to_anchor = (1, 1), loc='upper left',frameon = False)

    fig.subplots_adjust(wspace=0.2,hspace = 0.5,top= 0.975, bottom = 0.145, left = 0.08,right = 0.685)
  

    
def PlotRecVSKappaEQMICOD_SUBPLOTSFINAL():
    pthfolder = './sequences/phylogeny_K1_100_alpha90_M5_100real/'
    pthfolder2 = './sequences/phylogeny_K1_100_alpha90_M15_100real/'
    
    ptheq = './sequences/EQ_K1_100_alpha90_100real/'
    parametername='Kappa'
    valname1 = 'Alpha'
    valname2 = 'Mutations'
    pseudocount = 0.00001

    pthfolder2 = pthfolder+'/2023_04_25_11_02_04_Sectors_nspins200_flips3000_seed_12_deltanormal_M5_alpha90_G11_realnbr0.h5'
    _,_,parameters,val1,val2 = OpenFile2(pthfolder2, parametername, valname1,valname2)
    
    parameters =parameters/4
    random_expectation = RandomExpectation(ptheq)
    
    lwdth = 2
    fig,axs = plt.subplots(1,2)

    
    pathm5 = './results/recovery/phylogeny_K1_100_alpha90_M5_100real/'
    C_lmax,C_lmin, icod_Lmax,icod_Lmin,_,_,_ = LoadRec(pathm5)
    
    pathmain = './results/recovery/phylogeny_K1_100_alpha90_M15_100real/'
    C_lmax2,C_lmin2, icod_Lmax2,icod_Lmin2,_,_,_ = LoadRec(pathmain)
    
    patheq = './results/recovery/EQ_K1_100_alpha90_100real/'
    C_lmaxeq,C_lmineq,icod_Lmaxeq,icod_Lmineq,_,_,_ = LoadRec(patheq)
    
    pathm5 = './results/recovery/phylogeny_K1_100_alpha140_M5_100real/'
    C_lmax_T140,C_lmin_T140, icod_Lmax_T140,icod_Lmin_T140,_,_,_ = LoadRec(pathm5)
    
    pathmain = './results/recovery/phylogeny_K1_100_alpha140_M15_100real/'
    C_lmax2_T140,C_lmin2_T140, icod_Lmax2_T140,icod_Lmin2_T140,_,_,_ = LoadRec(pathmain)
    
    patheq = './results/recovery/EQ_K1_100_alpha140_100real/'
    C_lmaxeq_T140,C_lmineq_T140,icod_Lmaxeq_T140,icod_Lmineq_T140,_,_,_ = LoadRec(patheq)

    
    axs[0].plot(parameters,icod_Lmax,'-o', color = newcolorspal[2],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[0].plot(parameters,icod_Lmax2,'-o', color = newcolorspal[1],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')
    axs[0].plot(parameters,icod_Lmaxeq,'-o', color = newcolorspal[0],linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    axs[0].plot(parameters,C_lmin,'--s', color = newcolorspal[2],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[0].plot(parameters,C_lmin2,'--s', color = newcolorspal[1],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')
    axs[0].plot(parameters,C_lmineq,'--s', color = newcolorspal[0],linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    axs[0].plot(parameters,random_expectation*np.ones(parameters.shape),':', color = 'black',linewidth = lwdth,label = 'Null model',zorder = -1)

    axs[1].plot(parameters,icod_Lmax_T140,'-o', color = newcolorspal[2],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[1].plot(parameters,icod_Lmax2_T140,'-o', color = newcolorspal[1],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')
    axs[1].plot(parameters,icod_Lmaxeq_T140,'-o', color = newcolorspal[0],linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    axs[1].plot(parameters,C_lmin_T140,'--s', color = newcolorspal[2],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[1].plot(parameters,C_lmin2_T140,'--s', color = newcolorspal[1],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')
    axs[1].plot(parameters,C_lmineq_T140,'--s', color = newcolorspal[0],linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    axs[1].plot(parameters,random_expectation*np.ones(parameters.shape),':', color = 'black',linewidth = lwdth,label = 'Null model',zorder = -1)

    axs[0].set_title(r'$\mathbf{T^{*} = 90}$')
    axs[1].set_title(r'$\mathbf{T^{*} = 140}$')
    
    axs[0].set_ylabel('Recovery')
    axs[0].set_xlabel(r'Rescaled selection strength $\tilde{\kappa}$')
    axs[1].set_xlabel(r'Rescaled selection strength $\tilde{\kappa}$')
    
    axs[0].set_xlim([-1.5/4,101.5/4])
    axs[1].set_xlim([-1.5/4,101.5/4])
    
    custom_lines = ["ICOD $\Lambda_{max}$: ",
                    Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '-',marker = 'o'),
                    Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '-',marker = 'o'),
                    Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '-',marker = 'o'),
                    '',
                    "C $\lambda_{min}$:",
                    Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '--',marker = 's'),
                    Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '--',marker = 's'),
                    Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '--',marker = 's'),
                    '',
                    Line2D([0], [0], color='black', lw=lwdth,linestyle = ':')]
    
    labels = ['','No phylogeny','$\mu = 15$','$\mu = 5$','','','No phylogeny','$\mu = 15$','$\mu = 5$','','Null model']#,'$\mu = 5$ + APC','$\mu = 15$ + APC','eq + APC' ]
    
    
    axs[1].legend(custom_lines,labels,bbox_to_anchor = (1.05, 1.15), loc='upper left')

    fig.set_figheight(6.22)
    fig.set_figwidth(15.4)
    fig.subplots_adjust(wspace=0.18,hspace = 0.2,top= 0.9, bottom = 0.165, left = 0.075,right = 0.745)

     
def PlotRecAUCVSKappaEQMICOD_SUBPLOTSFINAL_SCA():
    pthfolder = './sequences/phylogeny_K1_100_alpha90_M5_100real/'
    pthfolder2 = './sequences/phylogeny_K1_100_alpha90_M15_100real/'
    
    ptheq = './sequences/EQ_K1_100_alpha90_100real/'
    parametername='Kappa'
    valname1 = 'Alpha'
    valname2 = 'Mutations'
    pseudocount = 0.00001

    pthfolder2 = pthfolder+'/2023_04_25_11_02_04_Sectors_nspins200_flips3000_seed_12_deltanormal_M5_alpha90_G11_realnbr0.h5'
    _,_,parameters,val1,val2 = OpenFile2(pthfolder2, parametername, valname1,valname2)

    parameters =parameters/4
    random_expectation = RandomExpectation(ptheq)

    
    pathm5 = './results/recovery/phylogeny_K1_100_alpha90_M5_100real/'
    C_lmax,C_lmin, icod_Lmax,icod_Lmin,sca_lmax,sca_lmin,cons = LoadRec(pathm5)
    
    pathmain = './results/recovery/phylogeny_K1_100_alpha90_M15_100real/'
    C_lmax2,C_lmin2, icod_Lmax2,icod_Lmin2,sca_lmax2,sca_lmin2,cons2 = LoadRec(pathmain)
    
    patheq = './results/recovery/EQ_K1_100_alpha90_100real/'
    C_lmaxeq,C_lmineq,icod_Lmaxeq,icod_Lmineq,sca_lmaxeq,sca_lmineq,conseq = LoadRec(patheq)
    
    pathm5 = './results/recovery/phylogeny_K1_100_alpha140_M5_100real/'
    C_lmax_T140,C_lmin_T140, icod_Lmax_T140,icod_Lmin_T140,sca_lmax_T140,sca_lmin_T140,cons_T140 = LoadRec(pathm5)
    
    pathmain = './results/recovery/phylogeny_K1_100_alpha140_M15_100real/'
    C_lmax2_T140,C_lmin2_T140, icod_Lmax2_T140,icod_Lmin2_T140,sca_lmax2_T140,sca_lmin2_T140,cons2_T140 = LoadRec(pathmain)
    
    patheq = './results/recovery/EQ_K1_100_alpha140_100real/'
    C_lmaxeq_T140,C_lmineq_T140,icod_Lmaxeq_T140,icod_Lmineq_T140,sca_lmaxeq_T140,sca_lmineq_T140,conseq_T140 = LoadRec(patheq)

    
    lwdth = 2
    fig,axs = plt.subplots(1,2, figsize=(15.4, 6.22))
    axs[0].plot(parameters,icod_Lmax,'-', color = newcolorspal[2],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[0].plot(parameters,icod_Lmax2,'-', color = newcolorspal[1],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')
    axs[0].plot(parameters,icod_Lmaxeq,'-', color = newcolorspal[0],linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    axs[0].plot(parameters,sca_lmax,'-s', color = newcolorspal[2],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[0].plot(parameters,sca_lmax2,'-s', color = newcolorspal[1],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')
    axs[0].plot(parameters,sca_lmaxeq,'-s', color = newcolorspal[0],linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    
    axs[0].plot(parameters,cons,'--d', color = newcolorspal[2],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[0].plot(parameters,cons2,'--d', color = newcolorspal[1],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')
    axs[0].plot(parameters,conseq,'--d', color = newcolorspal[0],linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    
    
    axs[0].plot(parameters,random_expectation*np.ones(parameters.shape),':', color = 'black',linewidth = lwdth,label = 'Null model',zorder = -1)

    axs[1].plot(parameters,icod_Lmax_T140,'-', color = newcolorspal[2],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[1].plot(parameters,icod_Lmax2_T140,'-', color = newcolorspal[1],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')
    axs[1].plot(parameters,icod_Lmaxeq_T140,'-', color = newcolorspal[0],linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    axs[1].plot(parameters,sca_lmax_T140,'-s', color = newcolorspal[2],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[1].plot(parameters,sca_lmax2_T140,'-s', color = newcolorspal[1],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')
    axs[1].plot(parameters,sca_lmaxeq_T140,'-s', color = newcolorspal[0],linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    
    axs[1].plot(parameters,cons_T140,'--d', color = newcolorspal[2],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[1].plot(parameters,cons2_T140,'--d', color = newcolorspal[1],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')
    axs[1].plot(parameters,conseq_T140,'--d', color = newcolorspal[0],linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    
    
    
    axs[1].plot(parameters,random_expectation*np.ones(parameters.shape),':', color = 'black',linewidth = lwdth,label = 'Null model',zorder = -1)
    axs[0].set_title(r'$\mathbf{\tau^{*} = 90}$')
    axs[1].set_title(r'$\mathbf{\tau^{*} = 140}$')
    axs[0].set_ylabel('Recovery')
    axs[0].set_xlabel(r'Rescaled selection strength $\tilde{\kappa}$')
    axs[1].set_xlabel(r'Rescaled selection strength $\tilde{\kappa}$')
    
    axs[0].set_xlim([-1.5/4,101.5/4])
    axs[1].set_xlim([-1.5/4,101.5/4])
    

    mks = 6
    line1eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '-' ,markersize=mks)
    line3eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '-',marker = 's',markersize=mks)
    line4eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '--',marker = 'd',markersize=mks)
    
    line1m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '-',markersize=mks)
    line3m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '-',marker = 's',markersize=mks)
    line4m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '--',marker = 'd',markersize=mks)

    line1m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '-',markersize=mks)
    line3m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '-',marker = 's',markersize=mks)
    line4m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '--',marker = 'd',markersize=mks)

    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

    label_j_1 = [""]
    label_j_2 = ["No phylogeny"]
    label_j_3 = ["$\mu = 15$"]
    label_j_4 = ["$\mu = 5$"]
    label_empty = [""]

    label_mthd1 = ["ICOD $\Lambda_{max}$"]

    label_mthd3 = ["SCA $\lambda_{max}$"]
    label_mthd4 = ["Conservation"]
    
    legend_handle = [extra,extra,extra,extra,
                     extra,line1eq,line3eq,line4eq,
                     extra,line1m15,line3m15,line4m15,
                     extra,line1m5,line3m5,line4m5]

    legend_labels = np.concatenate([label_empty,label_mthd1,label_mthd3,label_mthd4,
                                         label_j_2,label_empty * 3,
                                         label_j_3,label_empty * 3,
                                         label_j_4,label_empty *3,
                                         ])



    l2 = plt.legend([extra,Line2D([0], [0], color='black', lw=lwdth,linestyle = ':')],['Null model',''],ncol = 2, columnspacing = 1.7,handletextpad = -2,handleheight=1.5,handlelength=2,labelspacing=0.5,bbox_to_anchor = (0.91, 0.18),fontsize=fontsize_test, loc='upper left',frameon = False)
    
    for isxt,t in enumerate(l2.get_texts()):

        t.set_ha('left')

        if t.get_text() == "Null model":
            t.set_position((30,0))
            
    axs[1].add_artist(l2)

    l = axs[1].legend(legend_handle, legend_labels, ncol = 4,columnspacing = 1.7,handletextpad = -2,handleheight=1.5,handlelength=2,labelspacing=0.5,bbox_to_anchor = (0.91, 1.15),fontsize=fontsize_test, loc='upper left',frameon = False)#,  #bbox_to_anchor = (0.5,0.7),loc='upper left')

    for isxt,t in enumerate(l.get_texts()):
        t.set_ha('left')

        if t.get_text() == 'No phylogeny':

            t.set_ha('left')
            t.set_position((15,-10))
            t.set_rotation(90)

        if t.get_text() == '$\mu = 15$':
            t.set_ha('left')

            t.set_position((15,-10))
            t.set_rotation(90)
            
        if t.get_text() == '$\mu = 5$':
            t.set_ha('left')

            t.set_position((15,-10))
            t.set_rotation(90)
            
        if t.get_text() == "ICOD $\Lambda_{max}$":

            t.set_position((30,0))

        if t.get_text() == "C $\lambda_{min}$":

            t.set_position((30,0))

        if t.get_text() ==  "SCA $\lambda_{max}$":

            t.set_position((30,0))

        if t.get_text() == "Conservation":

            t.set_position((30,0))

        if t.get_text() =='Left panel:':
            t.set_ha('left')

    

    fig.subplots_adjust(wspace=0.18,hspace = 0.2,top= 0.9, bottom = 0.165, left = 0.075,right = 0.685)
  
def PlotExtremeTSpectrumEV():
    pthfolder = './sequences/phylogeny_Alpham300_300_K40_M5/'
    pthfolder2 = './sequences/phylogeny_Alpham300_300_K40_M15/'
    ptheq = './sequences/EQ_K40_alpham300_300/'
    parametername='Alpha'
    valname1 = 'Kappa'
    valname2 = 'Mutations'
    pseudocount = 0.00001
    
    pthtmp = './sequences/phylogeny_Alpham300_300_K40_M5_100real/2023_04_24_15_27_11_Sectors_nspins200_flips3000_seed_12_deltanormal_M5_Kappa40_G11_realnbr0.h5'
    _,_,parameters,val1,val2 = OpenFile2(pthtmp, parametername, valname1,valname2)
    
    random_expect = RandomExpectation('./sequences/phylogeny_Alpham300_300_K40_M5_100real/')
    
    icod_Lmin_5m = np.load('./results/recovery/phylogeny_Alpham300_300_K40_M5_100real/ICOD_Lmin.npy')
    C_lmax_5m = np.load('./results/recovery/phylogeny_Alpham300_300_K40_M5_100real/C_lmax.npy')
    
    icod_Lmin_15m = np.load('./results/recovery/phylogeny_Alpham300_300_K40_M15_100real/ICOD_Lmin.npy')
    C_lmax_15m = np.load('./results/recovery/phylogeny_Alpham300_300_K40_M15_100real/C_lmax.npy')
    
    icod_Lmin_eq = np.load('./results/recovery/EQ_K40_alpham300_300_100real/ICOD_Lmin.npy')
    C_lmax_eq = np.load('./results/recovery/EQ_K40_alpham300_300_100real/C_lmax.npy')
    
    
    filepath_eq = './sequences/EQ_K40_alpham300_300/2022_12_06_18_54_13_Sectors_nspins200_flips3000_nseq2048_seed_76_deltanormal_Alphamin-300_Alphamax300_K40_realnbr0.h5'

    
    eigvalC_t90,eigvalICOD_t90,pval1_t90,pval2_t90 = Eigenvalues(filepath_eq,'Alpha2',90,'Kappa',pseudocount)
    eigvalC_t195,eigvalICOD_t195,pval1_t195,pval2_t195 = Eigenvalues(filepath_eq,'Alpha2',195,'Kappa',pseudocount)
    eigvalC_t300,eigvalICOD_t300,pval1_t300,pval2_t300 = Eigenvalues(filepath_eq,'Alpha2',300,'Kappa',pseudocount)
    
    ev_cov_t90, egicod_t90,val1_t90,val2_t90 = Eigenvectors(filepath_eq,'Alpha2',90,'Kappa',pseudocount)
    ev_cov_t195, egicod_t195,val1_t195,val2_t195 = Eigenvectors(filepath_eq,'Alpha2',195,'Kappa',pseudocount)
    ev_cov_t300, egicod_t300,val1_t300,val2_t300 = Eigenvectors(filepath_eq,'Alpha2',300,'Kappa',pseudocount)
    
    delta = np.load('./vector_mut_effects_2states/delta.npy')
    delta = delta/2
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize = (16,7.2))
    # fig,axs = plt.subplots(2,2)
    mksize = 6
    
    
    ax = plt.subplot(gs[0, 0])
    lwdth = 2
    rank = np.arange(eigvalICOD_t90.shape[0])
    ax.plot(rank,eigvalICOD_t90,'-o',color = newcolorspal[0],markerfacecolor="None",markeredgewidth=1.5,markersize = mksize,label = '$T^* = 90$')
    ax.set_ylabel('Eigenvalues')

    ax.set_xlim([-2,202])
    ax.set_xlabel('Rank')
    ax = plt.subplot(gs[0, 1])
    lwdth = 2
    rank = np.arange(eigvalICOD_t90.shape[0])
    ax.plot(rank,eigvalICOD_t195,'-^',color = newcolorspal[5],markerfacecolor="None",markeredgewidth=1.5,markersize = mksize,label = '$T^* = 195$')
    ax.plot(rank,eigvalICOD_t300,'-s',color = newcolorspal[4],markerfacecolor="None",markeredgewidth=1.5,markersize = mksize,label = '$T^* = 300$')

    ax.set_ylabel('Eigenvalues')
    ax.ticklabel_format(axis ='y',style = 'sci',scilimits=(0,0))
    ax.set_xlim([-2,202])

    left, bottom, width, height = [0.55, 0.69, 0.15, 0.12]
    ax3 = fig.add_axes([left, bottom, width, height])
    cutoff_xscale = 25
    ax3.plot(rank[:cutoff_xscale],eigvalICOD_t195[:cutoff_xscale],'-^',color = newcolorspal[5],markerfacecolor="None",markeredgewidth=1.5,markersize = mksize,label = '$T^* = 195$')
    ax3.plot(rank[:cutoff_xscale],eigvalICOD_t300[:cutoff_xscale],'-s',color = newcolorspal[4],markerfacecolor="None",markeredgewidth=1.5,markersize = mksize,label = '$T^* = 300$')
 
    ax3.set_xlim([-0.5,cutoff_xscale-0.8])
    ax3.set_ylim([-3000,26000])
    ax3.set_xticks([0,10,20])
    ax3.ticklabel_format(axis ='y',style = 'sci',scilimits=(0,0))
    lwdth = 2
    custom_lines = [
                    Line2D([0], [0], color=newcolorspal[0], lw=0,markerfacecolor="None",markeredgewidth=1.5,marker = 'o'),
                    Line2D([0], [0], color=newcolorspal[5], lw=0,markerfacecolor="None",markeredgewidth=1.5,marker = '^'),
                    Line2D([0], [0], color=newcolorspal[4], lw=0,markerfacecolor="None",markeredgewidth=1.5,marker = 's')
                    ]
    
    labels = [r'$\tau^* = 90$',r'$\tau^* = 195$',r'$\tau^* = 300$']
    ax.legend(custom_lines,labels,bbox_to_anchor = (1.05, 1), loc='upper left',prop={'size': fontsize_test})

    ax.set_xlabel('Rank')
    
    ctf = 50
    idxs_ctfs = rank <= ctf
    rank = rank[idxs_ctfs]

    ax = plt.subplot(gs[1, 1])
    ax.plot(rank,egicod_t195[idxs_ctfs,-1]/np.linalg.norm(egicod_t195[idxs_ctfs,-1]),'^',markerfacecolor="None",color = newcolorspal[5],markeredgewidth=1.5,markersize = mksize)

    ax.plot(rank,egicod_t90[idxs_ctfs,-1]/np.linalg.norm(egicod_t90[idxs_ctfs,-1])*-1,'o',color = newcolorspal[0],markerfacecolor="None",markeredgewidth=1.5,markersize = mksize)
    ax.plot(rank,egicod_t300[idxs_ctfs,-1]/np.linalg.norm(egicod_t300[idxs_ctfs,-1])*-1,'s',color = newcolorspal[4],markerfacecolor="None",markeredgewidth=1.5,markersize = mksize)
    ax.plot(rank,delta[idxs_ctfs]/np.linalg.norm(delta[idxs_ctfs]),color = newcolorspal[1],label = 'Normalised'+r' $\vec{\Delta}$')
    ax.set_ylabel('Eigenvector component')
    # ax.set_ylabel('Eigenvector component')
    ax.set_xlabel('Site')
    ax.set_xlim([-2,ctf+2])
    ax.set_ylim([-0.8,0.8])
    custom_lines = [
                    Line2D([0], [0], color=newcolorspal[0], lw=0,markerfacecolor="None",markeredgewidth=1.5,marker = 'o'),
                    Line2D([0], [0], color=newcolorspal[5], lw=0,markerfacecolor="None",markeredgewidth=1.5,marker = '^'),
                    Line2D([0], [0], color=newcolorspal[4], lw=0,markerfacecolor="None",markeredgewidth=1.5,marker = 's'),
                    Line2D([0],[0], color = newcolorspal[1],lw = lwdth, linestyle = '-')]
    
    labels = [r'$\tau^* = 90$',r'$\tau^* = 195$',r'$\tau^* = 300$','Normalised'+r' $\vec{D}$']

    ax.legend(custom_lines,labels,bbox_to_anchor = (1.05, 1), loc='upper left',prop={'size': fontsize_test})
    ax.text(40,0.5,'$\Lambda_{min}$')
    # ax.set_yticks([])
    ax = plt.subplot(gs[1, 0])
    ax.plot(rank,egicod_t195[idxs_ctfs,0]/np.linalg.norm(egicod_t195[idxs_ctfs,0]),'^',markerfacecolor="None",color = newcolorspal[5],markeredgewidth=1.5,markersize = mksize)

    ax.plot(rank,egicod_t90[idxs_ctfs,0]/np.linalg.norm(egicod_t90[idxs_ctfs,0])*-1,'o',color = newcolorspal[0],markerfacecolor="None",markeredgewidth=1.5,markersize = mksize)
    ax.plot(rank,egicod_t300[idxs_ctfs,0]/np.linalg.norm(egicod_t300[idxs_ctfs,0])*-1,'s',color = newcolorspal[4],markerfacecolor="None",markeredgewidth=1.5,markersize = mksize)
    ax.plot(rank,delta[idxs_ctfs]/np.linalg.norm(delta[idxs_ctfs]),color = newcolorspal[1],label = 'Normalised'+r' $\vec{\Delta}$')
 
    ax.set_ylabel('Eigenvector component')
    ax.set_xlabel('Site')
    ax.set_xlim([-2,ctf+2])
    ax.set_ylim([-0.8,0.8])
    ax.text(40,0.5,'$\Lambda_{max}$')

    fig.subplots_adjust(wspace=0.455,hspace = 0.5,top= 0.946, bottom = 0.123, left = 0.087,right = 0.765)
    

def Conservation(finalchains):
    conservation =[]
    nbrseq, nbrsites = finalchains.shape
    # nbrstates = np.unique(finalchains).shape[0]
    # freq = np.zeros((nbrstates,nbrsites))
    h_entropy = np.zeros(nbrsites)
    for j in range(0,nbrsites):
        val,cts = np.unique(finalchains[:,j],return_counts = True)
        # freq[:,j] = cts/nbrseq
        h_entropy[j] = entropy(cts/nbrseq,base = 2)
    

    conservation_singlesite = 1 - h_entropy
    
    return conservation_singlesite


def PlotRECVSAlphaEQM2panels():
    pthfolder = './sequences/phylogeny_Alpham300_300_K40_M5_100real/'
    pthfolder2 = './sequences/phylogeny_Alpham300_300_K40_M15_100real/'
    ptheq = './sequences/EQ_K40_alpham300_300_100real/'
    parametername='Alpha'
    valname1 = 'Kappa'
    valname2 = 'Mutations'
    pseudocount = 0.00001

    pathparam = './sequences/phylogeny_Alpham300_300_K40_M5_100real/2023_04_24_15_27_11_Sectors_nspins200_flips3000_seed_12_deltanormal_M5_Kappa40_G11_realnbr0.h5'
    _,delta,parameters,val1,val2 = OpenFile2(pathparam, parametername, valname1,valname2)
    
    random_expectation = RandomExpectation(pthfolder)

    pathmain = './results/recovery/EQ_K40_alpham300_300_100real/'
    C_lmaxeq = np.load(pathmain+'C_lmax.npy')
    C_lmineq = np.load(pathmain+'C_lmin.npy')
    SCA_lmaxeq = np.load(pathmain+'SCA_lmax.npy')
    SCA_lmineq = np.load(pathmain+'SCA_lmin.npy')
    icod_Lmaxeq = np.load(pathmain+'ICOD_Lmax.npy')
    icod_Lmineq = np.load(pathmain+'ICOD_Lmin.npy')
    conservationeq = np.load(pathmain+'conservation.npy')
    

    
    pathmain = './results/recovery/phylogeny_Alpham300_300_K40_M5_100real/'
    C_lmax = np.load(pathmain+'C_lmax.npy')
    C_lmin = np.load(pathmain+'C_lmin.npy')
    SCA_lmax = np.load(pathmain+'SCA_lmax.npy')
    SCA_lmin = np.load(pathmain+'SCA_lmin.npy')
    icod_Lmax = np.load(pathmain+'ICOD_Lmax.npy')
    icod_Lmin = np.load(pathmain+'ICOD_Lmin.npy')
    conservation = np.load(pathmain+'conservation.npy')

    
    
    pathmain = './results/recovery/phylogeny_Alpham300_300_K40_M15_100real/'
    C_lmax2 = np.load(pathmain+'C_lmax.npy')
    C_lmin2 = np.load(pathmain+'C_lmin.npy')
    SCA_lmax2 = np.load(pathmain+'SCA_lmax.npy')
    SCA_lmin2 = np.load(pathmain+'SCA_lmin.npy')
    icod_Lmax2 = np.load(pathmain+'ICOD_Lmax.npy')
    icod_Lmin2 = np.load(pathmain+'ICOD_Lmin.npy')
    conservation2 = np.load(pathmain+'conservation.npy')
    
    lwdth = 2
    fig,axs = plt.subplots(1,2,figsize = (16,7.16))

    idxs = parameters>=0

    mks = 6
    axs[0].plot(parameters[idxs],icod_Lmax[idxs],'-o', color = newcolorspal[2], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[0].plot(parameters[idxs],C_lmin[idxs],'--^',color = newcolorspal[2], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, \mu = 5')

    axs[1].plot(parameters[idxs],SCA_lmax[idxs],'-s',color = newcolorspal[2],markersize=mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, \mu = 5')
    
    axs[0].plot(parameters[idxs],icod_Lmax2[idxs],'-o', color = newcolorspal[1], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')
    axs[0].plot(parameters[idxs],C_lmin2[idxs],'--^',color = newcolorspal[1], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}, \mu = 15$')
    axs[1].plot(parameters[idxs],SCA_lmax2[idxs],'-s',color = newcolorspal[1],markersize=mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, \mu = 5')
    
    axs[0].plot(parameters[idxs],icod_Lmaxeq[idxs],'-o', color = newcolorspal[0], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    axs[0].plot(parameters[idxs],C_lmineq[idxs],'--^',color = newcolorspal[0], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, EQ')
    axs[1].plot(parameters[idxs],SCA_lmaxeq[idxs],'-s',color = newcolorspal[0],markersize=mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, \mu = 5')
    
    axs[1].plot(parameters[idxs],conservationeq[idxs],'--d', markersize = mks, linewidth = lwdth,color = newcolorspal[0])
    axs[1].plot(parameters[idxs],conservation2[idxs],'--d', markersize = mks,linewidth = lwdth,color = newcolorspal[1])
    axs[1].plot(parameters[idxs],conservation[idxs],'--d', markersize = mks,linewidth = lwdth,color = newcolorspal[2])
    
    
    axs[0].plot(parameters[idxs],random_expectation*np.ones(np.sum(idxs)),':',color = 'black',linewidth = lwdth,zorder = -1)
    axs[1].plot(parameters[idxs],random_expectation*np.ones(np.sum(idxs)),':',color = 'black',linewidth = lwdth,zorder = -1)
    
    axs[0].set_ylabel('Recovery')

    axs[0].set_xlabel(r'Favored trait value $\tau^{*}$')
    axs[1].set_xlabel(r'Favored trait value $\tau^{*}$')
    
    line1eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '-',marker = 'o' ,markersize=mks)
    line2eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '--',marker = '^',markersize=mks)
    line3eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '-',marker = 's',markersize=mks)
    line4eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '--',marker = 'd',markersize=mks)
    
    line1m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '-',marker = 'o',markersize=mks)
    line2m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '--',marker = '^',markersize=mks)
    line3m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '-',marker = 's',markersize=mks)
    line4m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '--',marker = 'd',markersize=mks)

    line1m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '-',marker = 'o',markersize=mks)
    line2m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '--',marker = '^',markersize=mks)
    line3m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '-',marker = 's',markersize=mks)
    line4m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '--',marker = 'd',markersize=mks)

    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

    label_j_1 = [""]
    label_j_2 = ["No phylogeny"]
    label_j_3 = ["$\mu = 15$"]
    label_j_4 = ["$\mu = 5$"]
    label_empty = [""]
    
    label_panel1 = " \n Left panel:"
    label_panel2 = " \n Right panel:"
    
    label_mthd1 = ["ICOD $\Lambda_{max}$"]
    label_mthd2 = ["C $\lambda_{min}$"]
    label_mthd3 = ["SCA $\lambda_{max}$"]
    label_mthd4 = ["Conservation"]
    
    legend_handle = [extra,label_panel1,extra,extra,label_panel2,extra,extra,extra,extra,line1eq, line2eq, extra,line3eq,line4eq,
                     extra,extra,line1m15, line2m15, extra,line3m15,line4m15,
                     extra,extra,line1m5, line2m5, extra,line3m5,line4m5]

    legend_labels = np.concatenate([label_empty, label_empty,label_mthd1,label_mthd2,label_empty,label_mthd3,label_mthd4,
                                         label_j_2,label_empty * 6,
                                         label_j_3,label_empty * 6,
                                         label_j_4,label_empty * 6,
                                         ])


    l2 = plt.legend([ extra,Line2D([0], [0], color='black', lw=lwdth,linestyle = ':')],[' Null model',''],ncol = 2, columnspacing = 0.8,handletextpad = -2,handleheight=1.5,handlelength=2,labelspacing=0.5,bbox_to_anchor = (0.98, 0.18),fontsize=fontsize_test, loc='upper left',frameon = False)
    
    for isxt,t in enumerate(l2.get_texts()):

        t.set_ha('left')

    axs[1].add_artist(l2)

    
    l = axs[1].legend(legend_handle, legend_labels, ncol = 4,columnspacing = 1.7,handletextpad = -2,handleheight=1.5,handlelength=2,labelspacing=0.5,bbox_to_anchor = (0.98, 1.15),fontsize=fontsize_test, loc='upper left',frameon = False)#,  #bbox_to_anchor = (0.5,0.7),loc='upper left')

    for isxt,t in enumerate(l.get_texts()):
        t.set_ha('left')

        if t.get_text() == 'No phylogeny':

            t.set_ha('left')
            t.set_position((15,-50))
            t.set_rotation(90)

        if t.get_text() == '$\mu = 15$':
            t.set_ha('left')
            t.set_position((15,-50))
            t.set_rotation(90)
            
        if t.get_text() == '$\mu = 5$':
            t.set_ha('left')
            t.set_position((15,-50))
            t.set_rotation(90)
            
        if t.get_text() == "ICOD $\Lambda_{max}$":
            t.set_position((28,0))

        if t.get_text() == "C $\lambda_{min}$":
            t.set_position((28,0))

        if t.get_text() ==  "SCA $\lambda_{max}$":
            t.set_position((28,0))

        if t.get_text() == "Conservation":
            t.set_position((28,0))


    
    minval = np.min(parameters[idxs])-5
    maxval = np.max(parameters[idxs])+5
    
    axs[1].set_xlim([minval,maxval])
    axs[0].set_xlim([minval,maxval])

    axs[1].set_ylim([0.07,1.01])
    axs[0].set_ylim([0.07,1.01])
    fig.subplots_adjust(wspace=0.2,hspace = 0.5,top= 0.975, bottom = 0.145, left = 0.065,right = 0.680)

def PlotAUCVSAlphaEQM2panels():
    pthfolder = './sequences/phylogeny_Alpham300_300_K40_M5_100real/'
    pthfolder2 = './sequences/phylogeny_Alpham300_300_K40_M15_100real/'
    ptheq = './sequences/EQ_K40_alpham300_300_100real/'
    parametername='Alpha'
    valname1 = 'Kappa'
    valname2 = 'Mutations'
    pseudocount = 0.00001

    pathparam = './sequences/phylogeny_Alpham300_300_K40_M5_100real/2023_04_24_15_27_11_Sectors_nspins200_flips3000_seed_12_deltanormal_M5_Kappa40_G11_realnbr0.h5'
    _,_,parameters,val1,val2 = OpenFile2(pathparam, parametername, valname1,valname2)
    
    # random_expectation = RandomExpectation(pthfolder)
    random_expectation = 0
    pathmain = './results/AUC_sym/EQ_K40_alpham300_300_100real/'
    C_lmaxeq = np.load(pathmain+'C_lmax.npy')
    C_lmineq = np.load(pathmain+'C_lmin.npy')
    SCA_lmaxeq = np.load(pathmain+'SCA_lmax.npy')
    SCA_lmineq = np.load(pathmain+'SCA_lmin.npy')
    
    icod_Lmaxeq = np.load(pathmain+'ICOD_Lmax.npy')
    icod_Lmineq = np.load(pathmain+'ICOD_Lmin.npy')
    conservationeq = np.load(pathmain+'conservation.npy')


    pathmain = './results/AUC_sym/phylogeny_Alpham300_300_K40_M5_100real/'
    C_lmax = np.load(pathmain+'C_lmax.npy')
    C_lmin = np.load(pathmain+'C_lmin.npy')
    SCA_lmax = np.load(pathmain+'SCA_lmax.npy')
    SCA_lmin = np.load(pathmain+'SCA_lmin.npy')
    icod_Lmax = np.load(pathmain+'ICOD_Lmax.npy')
    icod_Lmin = np.load(pathmain+'ICOD_Lmin.npy')
    conservation = np.load(pathmain+'conservation.npy')
    
    pathmain = './results/AUC_sym/phylogeny_Alpham300_300_K40_M15_100real/'
    C_lmax2 = np.load(pathmain+'C_lmax.npy')
    C_lmin2 = np.load(pathmain+'C_lmin.npy')
    SCA_lmax2 = np.load(pathmain+'SCA_lmax.npy')
    SCA_lmin2 = np.load(pathmain+'SCA_lmin.npy')
    icod_Lmax2 = np.load(pathmain+'ICOD_Lmax.npy')
    icod_Lmin2 = np.load(pathmain+'ICOD_Lmin.npy')
    conservation2 = np.load(pathmain+'conservation.npy')
    
    lwdth = 2
    fig,axs = plt.subplots(1,2,figsize = (16,7.16))
    # fig = plt.figure(figsize = (14.22,  7.16))
    # plt.title(r'N = 2048, $\kappa$={}'.format(val1))

    idxs = parameters>=0
    mks = 6
    axs[0].plot(parameters[idxs],icod_Lmax[idxs],'-o', color = newcolorspal[2], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[0].plot(parameters[idxs],C_lmin[idxs],'--^',color = newcolorspal[2], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, \mu = 5')

    # axs[1].plot(parameters[idxs],icod_Lmax[idxs],'-o', color = newcolorspal[2],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[1].plot(parameters[idxs],SCA_lmax[idxs],'-s',color = newcolorspal[2],markersize=mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, \mu = 5')
    
    axs[0].plot(parameters[idxs],icod_Lmax2[idxs],'-o', color = newcolorspal[1], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')
    axs[0].plot(parameters[idxs],C_lmin2[idxs],'--^',color = newcolorspal[1], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}, \mu = 15$')
    # axs[1].plot(parameters[idxs],icod_Lmax2[idxs],'-o', color = newcolorspal[1],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')  
    axs[1].plot(parameters[idxs],SCA_lmax2[idxs],'-s',color = newcolorspal[1],markersize=mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, \mu = 5')
    
    axs[0].plot(parameters[idxs],icod_Lmaxeq[idxs],'-o', color = newcolorspal[0], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    axs[0].plot(parameters[idxs],C_lmineq[idxs],'--^',color = newcolorspal[0], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, EQ')
    # axs[1].plot(parameters[idxs],icod_Lmaxeq[idxs],'-o', color = newcolorspal[0],linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    axs[1].plot(parameters[idxs],SCA_lmaxeq[idxs],'-s',color = newcolorspal[0],markersize=mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, \mu = 5')
    
    # axs[0].plot(parameters[idxs],conservationeq[idxs],'-P', markersize = 7,color = newcolorspal[0])
    axs[1].plot(parameters[idxs],conservationeq[idxs],'--d', markersize = mks, linewidth = lwdth,color = newcolorspal[0])
    # axs[0].plot(parameters[idxs],conservation2[idxs],'-P', markersize = 7, color = newcolorspal[1])
    axs[1].plot(parameters[idxs],conservation2[idxs],'--d', markersize = mks,linewidth = lwdth,color = newcolorspal[1])
    # axs[0].plot(parameters[idxs],conservation[idxs],'-P', markersize = 7, color = newcolorspal[2])
    axs[1].plot(parameters[idxs],conservation[idxs],'--d', markersize = mks,linewidth = lwdth,color = newcolorspal[2])

    
    axs[0].plot(parameters[idxs],random_expectation*np.ones(np.sum(idxs)),':',color = 'black',linewidth = lwdth,zorder = -1)
    axs[1].plot(parameters[idxs],random_expectation*np.ones(np.sum(idxs)),':',color = 'black',linewidth = lwdth,zorder = -1)
    
    axs[0].set_ylabel('Symmetrized AUC')
    # axs[0].set_xlabel(r'Favored trait value $T^{*}$')
    # axs[1].set_xlabel(r'Favored trait value $T^{*}$')
    axs[0].set_xlabel(r'Favored trait value $\tau^{*}$')
    axs[1].set_xlabel(r'Favored trait value $\tau^{*}$')

    line1eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '-',marker = 'o' ,markersize=mks)
    line2eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '--',marker = '^',markersize=mks)
    line3eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '-',marker = 's',markersize=mks)
    line4eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '--',marker = 'd',markersize=mks)
    
    line1m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '-',marker = 'o',markersize=mks)
    line2m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '--',marker = '^',markersize=mks)
    line3m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '-',marker = 's',markersize=mks)
    line4m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '--',marker = 'd',markersize=mks)

    line1m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '-',marker = 'o',markersize=mks)
    line2m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '--',marker = '^',markersize=mks)
    line3m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '-',marker = 's',markersize=mks)
    line4m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '--',marker = 'd',markersize=mks)


    

    # create blank rectangle
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    #Create organized list containing all handles for table. Extra represent empty space


    #Define the labels
    # label_row_1 = ["", "", "Phylogeny",'','']
    label_j_1 = [""]
    label_j_2 = ["No phylogeny"]
    label_j_3 = ["$\mu = 15$"]
    label_j_4 = ["$\mu = 5$"]
    label_empty = [""]
    
    label_panel1 = " \n Left panel:"
    label_panel2 = " \n Right panel:"
    
    label_mthd1 = ["ICOD $\Lambda_{max}$"]
    label_mthd2 = ["C $\lambda_{min}$"]
    label_mthd3 = ["SCA $\lambda_{max}$"]
    label_mthd4 = ["Conservation"]
    
    legend_handle = [extra,label_panel1,extra,extra,label_panel2,extra,extra,extra,extra,line1eq, line2eq, extra,line3eq,line4eq,
                     extra,extra,line1m15, line2m15, extra,line3m15,line4m15,
                     extra,extra,line1m5, line2m5, extra,line3m5,line4m5]

    legend_labels = np.concatenate([label_empty, label_empty,label_mthd1,label_mthd2,label_empty,label_mthd3,label_mthd4,
                                         label_j_2,label_empty * 6,
                                         label_j_3,label_empty * 6,
                                         label_j_4,label_empty * 6,
                                         ])


    #Create legend

    l2 = plt.legend([ extra,Line2D([0], [0], color='black', lw=lwdth,linestyle = ':')],[' Null model',''],ncol = 2, columnspacing = 0.8,handletextpad = -2,handleheight=1.5,handlelength=2,labelspacing=0.5,bbox_to_anchor = (0.98, 0.18),fontsize=fontsize_test, loc='upper left',frameon = False)
    
    for isxt,t in enumerate(l2.get_texts()):
    #     # if t.get_text() == 'Phylogeny':
    #     #     t.set_rotation(90)
    #     #     t.set_linespacing(0.01)
    #         # pass
        t.set_ha('left')
        # t.set_va('center')
        if t.get_text() == " Null model":
            # t.set_ha('left')
            pass
            # t.set_position((-30,0))
    axs[1].add_artist(l2)

    
    l = axs[1].legend(legend_handle, legend_labels, ncol = 4,columnspacing = 1.7,handletextpad = -2,handleheight=1.5,handlelength=2,labelspacing=0.5,bbox_to_anchor = (0.98, 1.15),fontsize=fontsize_test, loc='upper left',frameon = False)#,  #bbox_to_anchor = (0.5,0.7),loc='upper left')
    # pdb.set_trace()
    for isxt,t in enumerate(l.get_texts()):
        t.set_ha('left')
        # t.set_va('center')
        if t.get_text() == 'No phylogeny':

            # t.set_fontweight('normal')
            t.set_ha('left')
            t.set_position((15,-50))
            t.set_rotation(90)
            # pass
        if t.get_text() == '$\mu = 15$':
            # pass

            t.set_ha('left')
            # t.set_fontweight('normal')
            t.set_position((15,-50))
            t.set_rotation(90)
        if t.get_text() == '$\mu = 5$':
            # pass

            # t.set_color('red')
            t.set_ha('left')
            # t.set_fontweight('normal')
            t.set_position((15,-50))
            t.set_rotation(90)
        if t.get_text() == "ICOD $\Lambda_{max}$":

            # t.set_ha('left')
            # t.set_fontweight('normal')
            t.set_position((28,0))
            # t.set_rotation(90)
            pass
        if t.get_text() == "C $\lambda_{min}$":
            pass

            # t.set_ha('left')
            # t.set_fontweight('normal')
            t.set_position((28,0))
            # t.set_rotation(90)
        if t.get_text() ==  "SCA $\lambda_{max}$":
            pass

            # t.set_ha('left')
            # t.set_fontweight('normal')
            t.set_position((28,0))
            # t.set_rotation(90)
        if t.get_text() == "Conservation":
            pass

            # t.set_ha('right')
            # t.set_fontweight('normal')
            t.set_position((28,0))
            # t.set_rotation(90)
        if t.get_text() =='Left panel:':
            pass
 
            
            t.set_ha('left')
            # t.set_position((100,0))
            # t.set_fontweight('bold')
            # t.set_rotation(90)
        if t.get_text() == "Right panel:":
            pass
            
            # t.set_
            # t.set_ha('left')
            # t.set_position((100,0))
            # t.set_text('Right panel:')
            # t.set_fontweight('bold')
            # t.set_rotation(90)
            
    axs[1].set_xlim([-5,305])
    axs[0].set_xlim([-5,305])
    axs[1].set_ylim([-0.01,1.01])
    axs[0].set_ylim([-0.01,1.01])
    fig.subplots_adjust(wspace=0.2,hspace = 0.5,top= 0.975, bottom = 0.145, left = 0.065,right = 0.680)


def PlotRECVSAlphaEQM2panels_oppositeSPECTRUM():
    pthfolder = './sequences/phylogeny_Alpham300_300_K40_M5_100real/'
    pthfolder2 = './sequences/phylogeny_Alpham300_300_K40_M15_100real/'
    ptheq = './sequences/EQ_K40_alpham300_300_100real/'
    parametername='Alpha'
    valname1 = 'Kappa'
    valname2 = 'Mutations'
    pseudocount = 0.00001

    pathparam = './sequences/phylogeny_Alpham300_300_K40_M5_100real/2023_04_24_15_27_11_Sectors_nspins200_flips3000_seed_12_deltanormal_M5_Kappa40_G11_realnbr0.h5'
    _,_,parameters,val1,val2 = OpenFile2(pathparam, parametername, valname1,valname2)
    
    random_expectation = RandomExpectation(pthfolder)

    pathmain = './results/recovery/EQ_K40_alpham300_300_100real/'
    C_lmaxeq = np.load(pathmain+'C_lmax.npy')
    C_lmineq = np.load(pathmain+'C_lmin.npy')
    SCA_lmaxeq = np.load(pathmain+'SCA_lmax.npy')
    SCA_lmineq = np.load(pathmain+'SCA_lmin.npy')
    
    icod_Lmaxeq = np.load(pathmain+'ICOD_Lmax.npy')
    icod_Lmineq = np.load(pathmain+'ICOD_Lmin.npy')
    conservationeq = np.load(pathmain+'conservation.npy')
    
    pathmain = './results/recovery/phylogeny_Alpham300_300_K40_M5_100real/'
    C_lmax = np.load(pathmain+'C_lmax.npy')
    C_lmin = np.load(pathmain+'C_lmin.npy')
    SCA_lmax = np.load(pathmain+'SCA_lmax.npy')
    SCA_lmin = np.load(pathmain+'SCA_lmin.npy')
    icod_Lmax = np.load(pathmain+'ICOD_Lmax.npy')
    icod_Lmin = np.load(pathmain+'ICOD_Lmin.npy')
    conservation = np.load(pathmain+'conservation.npy')
    
    pathmain = './results/recovery/phylogeny_Alpham300_300_K40_M15_100real/'
    C_lmax2 = np.load(pathmain+'C_lmax.npy')
    C_lmin2 = np.load(pathmain+'C_lmin.npy')
    SCA_lmax2 = np.load(pathmain+'SCA_lmax.npy')
    SCA_lmin2 = np.load(pathmain+'SCA_lmin.npy')
    icod_Lmax2 = np.load(pathmain+'ICOD_Lmax.npy')
    icod_Lmin2 = np.load(pathmain+'ICOD_Lmin.npy')
    conservation2 = np.load(pathmain+'conservation.npy')
    
    lwdth = 2
    fig,axs = plt.subplots(1,2,figsize = (16,7.16))
    # fig = plt.figure(figsize = (14.22,  7.16))
    # plt.title(r'N = 2048, $\kappa$={}'.format(val1))

    idxs = parameters>=0
    mks = 6
    axs[0].plot(parameters[idxs],icod_Lmin[idxs],'-o', color = newcolorspal[2], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[0].plot(parameters[idxs],C_lmax[idxs],'--^',color = newcolorspal[2], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, \mu = 5')

    # axs[1].plot(parameters[idxs],icod_Lmax[idxs],'-o', color = newcolorspal[2],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[1].plot(parameters[idxs],SCA_lmin[idxs],'-s',color = newcolorspal[2],markersize=mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, \mu = 5')
    
    axs[0].plot(parameters[idxs],icod_Lmin2[idxs],'-o', color = newcolorspal[1], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')
    axs[0].plot(parameters[idxs],C_lmax2[idxs],'--^',color = newcolorspal[1], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}, \mu = 15$')
    # axs[1].plot(parameters[idxs],icod_Lmax2[idxs],'-o', color = newcolorspal[1],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')  
    axs[1].plot(parameters[idxs],SCA_lmin2[idxs],'-s',color = newcolorspal[1],markersize=mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, \mu = 5')
    
    axs[0].plot(parameters[idxs],icod_Lmineq[idxs],'-o', color = newcolorspal[0], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    axs[0].plot(parameters[idxs],C_lmaxeq[idxs],'--^',color = newcolorspal[0], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, EQ')
    # axs[1].plot(parameters[idxs],icod_Lmaxeq[idxs],'-o', color = newcolorspal[0],linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    axs[1].plot(parameters[idxs],SCA_lmineq[idxs],'-s',color = newcolorspal[0],markersize=mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, \mu = 5')
    
    # axs[0].plot(parameters[idxs],conservationeq[idxs],'-P', markersize = 7,color = newcolorspal[0])
    axs[1].plot(parameters[idxs],conservationeq[idxs],'--d', markersize = mks, linewidth = lwdth,color = newcolorspal[0])
    # axs[0].plot(parameters[idxs],conservation2[idxs],'-P', markersize = 7, color = newcolorspal[1])
    axs[1].plot(parameters[idxs],conservation2[idxs],'--d', markersize = mks,linewidth = lwdth,color = newcolorspal[1])
    # axs[0].plot(parameters[idxs],conservation[idxs],'-P', markersize = 7, color = newcolorspal[2])
    axs[1].plot(parameters[idxs],conservation[idxs],'--d', markersize = mks,linewidth = lwdth,color = newcolorspal[2])
    
    
    axs[0].plot(parameters[idxs],random_expectation*np.ones(np.sum(idxs)),':',color = 'black',linewidth = lwdth,zorder = -1)
    axs[1].plot(parameters[idxs],random_expectation*np.ones(np.sum(idxs)),':',color = 'black',linewidth = lwdth,zorder = -1)
    
    axs[0].set_ylabel('Recovery')

    axs[0].set_xlabel(r'Favored trait value $\tau^{*}$')
    axs[1].set_xlabel(r'Favored trait value $\tau^{*}$')

    
    
    line1eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '-',marker = 'o' ,markersize=mks)
    line2eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '--',marker = '^',markersize=mks)
    line3eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '-',marker = 's',markersize=mks)
    line4eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '--',marker = 'd',markersize=mks)
    
    line1m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '-',marker = 'o',markersize=mks)
    line2m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '--',marker = '^',markersize=mks)
    line3m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '-',marker = 's',markersize=mks)
    line4m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '--',marker = 'd',markersize=mks)

    line1m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '-',marker = 'o',markersize=mks)
    line2m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '--',marker = '^',markersize=mks)
    line3m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '-',marker = 's',markersize=mks)
    line4m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '--',marker = 'd',markersize=mks)


    

    # create blank rectangle
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    #Create organized list containing all handles for table. Extra represent empty space


    #Define the labels
    # label_row_1 = ["", "", "Phylogeny",'','']
    label_j_1 = [""]
    label_j_2 = ["No phylogeny"]
    label_j_3 = ["$\mu = 15$"]
    label_j_4 = ["$\mu = 5$"]
    label_empty = [""]
    
    label_panel1 = " \n Left panel:"
    label_panel2 = " \n Right panel:"
    
    label_mthd1 = ["ICOD $\Lambda_{min}$"]
    label_mthd2 = ["C $\lambda_{max}$"]
    label_mthd3 = ["SCA $\lambda_{min}$"]
    label_mthd4 = ["Conservation"]
    
    legend_handle = [extra,label_panel1,extra,extra,label_panel2,extra,extra,extra,extra,line1eq, line2eq, extra,line3eq,line4eq,
                     extra,extra,line1m15, line2m15, extra,line3m15,line4m15,
                     extra,extra,line1m5, line2m5, extra,line3m5,line4m5]

    legend_labels = np.concatenate([label_empty, label_empty,label_mthd1,label_mthd2,label_empty,label_mthd3,label_mthd4,
                                         label_j_2,label_empty * 6,
                                         label_j_3,label_empty * 6,
                                         label_j_4,label_empty * 6,
                                         ])


    #Create legend

    l2 = plt.legend([ extra,Line2D([0], [0], color='black', lw=lwdth,linestyle = ':')],[' Null model',''],ncol = 2, columnspacing = 0.8,handletextpad = -2,handleheight=1.5,handlelength=2,labelspacing=0.5,bbox_to_anchor = (0.98, 0.18),fontsize=fontsize_test, loc='upper left',frameon = False)
    
    for isxt,t in enumerate(l2.get_texts()):
    #     # if t.get_text() == 'Phylogeny':
    #     #     t.set_rotation(90)
    #     #     t.set_linespacing(0.01)
    #         # pass
        t.set_ha('left')
        # t.set_va('center')
        if t.get_text() == " Null model":
            # t.set_ha('left')
            pass
            # t.set_position((-30,0))
    axs[1].add_artist(l2)

    
    l = axs[1].legend(legend_handle, legend_labels, ncol = 4,columnspacing = 1.7,handletextpad = -2,handleheight=1.5,handlelength=2,labelspacing=0.5,bbox_to_anchor = (0.98, 1.15),fontsize=fontsize_test, loc='upper left',frameon = False)#,  #bbox_to_anchor = (0.5,0.7),loc='upper left')
    # pdb.set_trace()
    for isxt,t in enumerate(l.get_texts()):
        t.set_ha('left')
        # t.set_va('center')
        if t.get_text() == 'No phylogeny':

            # t.set_fontweight('normal')
            t.set_ha('left')
            t.set_position((15,-50))
            t.set_rotation(90)
            # pass
        if t.get_text() == '$\mu = 15$':
            # pass

            t.set_ha('left')
            # t.set_fontweight('normal')
            t.set_position((15,-50))
            t.set_rotation(90)
        if t.get_text() == '$\mu = 5$':
            # pass

            # t.set_color('red')
            t.set_ha('left')
            # t.set_fontweight('normal')
            t.set_position((15,-50))
            t.set_rotation(90)
        if t.get_text() == "ICOD $\Lambda_{min}$":

            # t.set_ha('left')
            # t.set_fontweight('normal')
            t.set_position((28,0))
            # t.set_rotation(90)
            pass
        if t.get_text() == "C $\lambda_{max}$":
            pass

            # t.set_ha('left')
            # t.set_fontweight('normal')
            t.set_position((28,0))
            # t.set_rotation(90)
        if t.get_text() ==  "SCA $\lambda_{min}$":
            pass

            # t.set_ha('left')
            # t.set_fontweight('normal')
            t.set_position((28,0))
            # t.set_rotation(90)
        if t.get_text() == "Conservation":
            pass

            # t.set_ha('right')
            # t.set_fontweight('normal')
            t.set_position((28,0))
            # t.set_rotation(90)
        if t.get_text() =='Left panel:':
            pass
 
            
            t.set_ha('left')
            # t.set_position((100,0))
            # t.set_fontweight('bold')
            # t.set_rotation(90)
        if t.get_text() == "Right panel:":
            pass
            
            # t.set_
            # t.set_ha('left')
            # t.set_position((100,0))
            # t.set_text('Right panel:')
            # t.set_fontweight('bold')
            # t.set_rotation(90)
            
    axs[1].set_xlim([-5,305])
    axs[0].set_xlim([-5,305])
    axs[1].set_ylim([-0.02,1.01])
    axs[0].set_ylim([-0.02,1.01])
    fig.subplots_adjust(wspace=0.2,hspace = 0.5,top= 0.975, bottom = 0.145, left = 0.065,right = 0.680)
    
def PlotAUCVSAlphaEQM2panels_oppositeSPECTRUM():
    pthfolder = './sequences/phylogeny_Alpham300_300_K40_M5_100real/'
    pthfolder2 = './sequences/phylogeny_Alpham300_300_K40_M15_100real/'
    ptheq = './sequences/EQ_K40_alpham300_300_100real/'
    parametername='Alpha'
    valname1 = 'Kappa'
    valname2 = 'Mutations'
    pseudocount = 0.00001

    pathparam = './sequences/phylogeny_Alpham300_300_K40_M5_100real/2023_04_24_15_27_11_Sectors_nspins200_flips3000_seed_12_deltanormal_M5_Kappa40_G11_realnbr0.h5'
    _,_,parameters,val1,val2 = OpenFile2(pathparam, parametername, valname1,valname2)
    
    random_expectation = 0

    pathmain = './results/AUC_sym/EQ_K40_alpham300_300_100real/'
    C_lmaxeq = np.load(pathmain+'C_lmax.npy')
    C_lmineq = np.load(pathmain+'C_lmin.npy')
    SCA_lmaxeq = np.load(pathmain+'SCA_lmax.npy')
    SCA_lmineq = np.load(pathmain+'SCA_lmin.npy')
    
    icod_Lmaxeq = np.load(pathmain+'ICOD_Lmax.npy')
    icod_Lmineq = np.load(pathmain+'ICOD_Lmin.npy')
    conservationeq = np.load(pathmain+'conservation.npy')
    
    pathmain = './results/AUC_sym/phylogeny_Alpham300_300_K40_M5_100real/'
    C_lmax = np.load(pathmain+'C_lmax.npy')
    C_lmin = np.load(pathmain+'C_lmin.npy')
    SCA_lmax = np.load(pathmain+'SCA_lmax.npy')
    SCA_lmin = np.load(pathmain+'SCA_lmin.npy')
    icod_Lmax = np.load(pathmain+'ICOD_Lmax.npy')
    icod_Lmin = np.load(pathmain+'ICOD_Lmin.npy')
    conservation = np.load(pathmain+'conservation.npy')
    
    pathmain = './results/AUC_sym/phylogeny_Alpham300_300_K40_M15_100real/'
    C_lmax2 = np.load(pathmain+'C_lmax.npy')
    C_lmin2 = np.load(pathmain+'C_lmin.npy')
    SCA_lmax2 = np.load(pathmain+'SCA_lmax.npy')
    SCA_lmin2 = np.load(pathmain+'SCA_lmin.npy')
    icod_Lmax2 = np.load(pathmain+'ICOD_Lmax.npy')
    icod_Lmin2 = np.load(pathmain+'ICOD_Lmin.npy')
    conservation2 = np.load(pathmain+'conservation.npy')
    
    lwdth = 2
    fig,axs = plt.subplots(1,2,figsize = (16,7.16))
    # fig = plt.figure(figsize = (14.22,  7.16))
    # plt.title(r'N = 2048, $\kappa$={}'.format(val1))

    idxs = parameters>=0
    mks = 6
    axs[0].plot(parameters[idxs],icod_Lmin[idxs],'-o', color = newcolorspal[2], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[0].plot(parameters[idxs],C_lmax[idxs],'--^',color = newcolorspal[2], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, \mu = 5')

    # axs[1].plot(parameters[idxs],icod_Lmax[idxs],'-o', color = newcolorspal[2],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 5$')
    axs[1].plot(parameters[idxs],SCA_lmin[idxs],'-s',color = newcolorspal[2],markersize=mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, \mu = 5')
    
    axs[0].plot(parameters[idxs],icod_Lmin2[idxs],'-o', color = newcolorspal[1], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')
    axs[0].plot(parameters[idxs],C_lmax2[idxs],'--^',color = newcolorspal[1], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}, \mu = 15$')
    # axs[1].plot(parameters[idxs],icod_Lmax2[idxs],'-o', color = newcolorspal[1],linewidth = lwdth,label = 'ICOD $\Lambda_{max}, \mu = 15$')  
    axs[1].plot(parameters[idxs],SCA_lmin2[idxs],'-s',color = newcolorspal[1],markersize=mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, \mu = 5')
    
    axs[0].plot(parameters[idxs],icod_Lmineq[idxs],'-o', color = newcolorspal[0], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    axs[0].plot(parameters[idxs],C_lmaxeq[idxs],'--^',color = newcolorspal[0], markersize = mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, EQ')
    # axs[1].plot(parameters[idxs],icod_Lmaxeq[idxs],'-o', color = newcolorspal[0],linewidth = lwdth,label = 'ICOD $\Lambda_{max}$, EQ')
    axs[1].plot(parameters[idxs],SCA_lmineq[idxs],'-s',color = newcolorspal[0],markersize=mks,linewidth = lwdth,label = 'ICOD $\Lambda_{min}$, \mu = 5')
    
    # axs[0].plot(parameters[idxs],conservationeq[idxs],'-P', markersize = 7,color = newcolorspal[0])
    axs[1].plot(parameters[idxs],conservationeq[idxs],'--d', markersize = mks, linewidth = lwdth,color = newcolorspal[0])
    # axs[0].plot(parameters[idxs],conservation2[idxs],'-P', markersize = 7, color = newcolorspal[1])
    axs[1].plot(parameters[idxs],conservation2[idxs],'--d', markersize = mks,linewidth = lwdth,color = newcolorspal[1])
    # axs[0].plot(parameters[idxs],conservation[idxs],'-P', markersize = 7, color = newcolorspal[2])
    axs[1].plot(parameters[idxs],conservation[idxs],'--d', markersize = mks,linewidth = lwdth,color = newcolorspal[2])

    axs[0].plot(parameters[idxs],random_expectation*np.ones(np.sum(idxs)),':',color = 'black',linewidth = lwdth,zorder = -1)
    axs[1].plot(parameters[idxs],random_expectation*np.ones(np.sum(idxs)),':',color = 'black',linewidth = lwdth,zorder = -1)
    
    axs[0].set_ylabel('Symmetrized AUC')
    # axs[0].set_xlabel(r'Favored trait value $T^{*}$')
    # axs[1].set_xlabel(r'Favored trait value $T^{*}$')
    axs[0].set_xlabel(r'Favored trait value $\tau^{*}$')
    axs[1].set_xlabel(r'Favored trait value $\tau^{*}$')

    
    
    line1eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '-',marker = 'o' ,markersize=mks)
    line2eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '--',marker = '^',markersize=mks)
    line3eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '-',marker = 's',markersize=mks)
    line4eq = Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '--',marker = 'd',markersize=mks)
    
    line1m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '-',marker = 'o',markersize=mks)
    line2m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '--',marker = '^',markersize=mks)
    line3m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '-',marker = 's',markersize=mks)
    line4m15 = Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '--',marker = 'd',markersize=mks)

    line1m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '-',marker = 'o',markersize=mks)
    line2m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '--',marker = '^',markersize=mks)
    line3m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '-',marker = 's',markersize=mks)
    line4m5 = Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '--',marker = 'd',markersize=mks)


    

    # create blank rectangle
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    #Create organized list containing all handles for table. Extra represent empty space


    #Define the labels
    # label_row_1 = ["", "", "Phylogeny",'','']
    label_j_1 = [""]
    label_j_2 = ["No phylogeny"]
    label_j_3 = ["$\mu = 15$"]
    label_j_4 = ["$\mu = 5$"]
    label_empty = [""]
    
    label_panel1 = " \n Left panel:"
    label_panel2 = " \n Right panel:"
    
    label_mthd1 = ["ICOD $\Lambda_{min}$"]
    label_mthd2 = ["C $\lambda_{max}$"]
    label_mthd3 = ["SCA $\lambda_{min}$"]
    label_mthd4 = ["Conservation"]
    
    legend_handle = [extra,label_panel1,extra,extra,label_panel2,extra,extra,extra,extra,line1eq, line2eq, extra,line3eq,line4eq,
                     extra,extra,line1m15, line2m15, extra,line3m15,line4m15,
                     extra,extra,line1m5, line2m5, extra,line3m5,line4m5]

    legend_labels = np.concatenate([label_empty, label_empty,label_mthd1,label_mthd2,label_empty,label_mthd3,label_mthd4,
                                         label_j_2,label_empty * 6,
                                         label_j_3,label_empty * 6,
                                         label_j_4,label_empty * 6,
                                         ])

    l2 = plt.legend([ extra,Line2D([0], [0], color='black', lw=lwdth,linestyle = ':')],[' Null model',''],ncol = 2, columnspacing = 0.8,handletextpad = -2,handleheight=1.5,handlelength=2,labelspacing=0.5,bbox_to_anchor = (0.98, 0.18),fontsize=fontsize_test, loc='upper left',frameon = False)
    
    for isxt,t in enumerate(l2.get_texts()):

        t.set_ha('left')

    axs[1].add_artist(l2)

    
    l = axs[1].legend(legend_handle, legend_labels, ncol = 4,columnspacing = 1.7,handletextpad = -2,handleheight=1.5,handlelength=2,labelspacing=0.5,bbox_to_anchor = (0.98, 1.15),fontsize=fontsize_test, loc='upper left',frameon = False)#,  #bbox_to_anchor = (0.5,0.7),loc='upper left')

    for isxt,t in enumerate(l.get_texts()):
        t.set_ha('left')

        if t.get_text() == 'No phylogeny':


            t.set_ha('left')
            t.set_position((15,-50))
            t.set_rotation(90)

        if t.get_text() == '$\mu = 15$':


            t.set_ha('left')

            t.set_position((15,-50))
            t.set_rotation(90)
        if t.get_text() == '$\mu = 5$':

            t.set_ha('left')

            t.set_position((15,-50))
            t.set_rotation(90)
        if t.get_text() == "ICOD $\Lambda_{min}$":

            t.set_position((28,0))

        if t.get_text() == "C $\lambda_{max}$":

            t.set_position((28,0))

        if t.get_text() ==  "SCA $\lambda_{min}$":

            t.set_position((28,0))

        if t.get_text() == "Conservation":


            t.set_position((28,0))
            # t.set_rotation(90)
        if t.get_text() =='Left panel:':
            pass
 
            
            t.set_ha('left')


            
    axs[1].set_xlim([-5,305])
    axs[0].set_xlim([-5,305])
    axs[1].set_ylim([-0.01,1.01])
    axs[0].set_ylim([-0.01,1.01])
    fig.subplots_adjust(wspace=0.2,hspace = 0.5,top= 0.975, bottom = 0.145, left = 0.065,right = 0.680)
    

def PlotEigenvaluesSuperpositionPurePhylogenySelection_INSET_subplotsSCA():

    filepath = './sequences/pure_phylogeny/2023_03_24_15_50_40_Sectors_nspins200_seed_12_G11_realnbr0.h5'
    pseudocount = 0.00001
    eigvalC,eigvalICOD,pval1,pval2 = Eigenvalues(filepath,'Mutations',50,'NumberGenerations',pseudocount)
    eigvalC_m15,eigvalICOD_m15,pval1,pval2 = Eigenvalues(filepath,'Mutations',15,'NumberGenerations',pseudocount)
    eigvalC_m5,eigvalICOD_m5,pval1,pval2 = Eigenvalues(filepath,'Mutations',5,'NumberGenerations',pseudocount)
    
    sca_m5,_,pval1,pval2 =EigenvaluesEigenvectorsSCA(filepath,'Mutations',5,'NumberGenerations',0)
    sca_m15,_,pval1,pval2 =EigenvaluesEigenvectorsSCA(filepath,'Mutations',15,'NumberGenerations',0)
    sca_m50,_,pval1,pval2 =EigenvaluesEigenvectorsSCA(filepath,'Mutations',50,'NumberGenerations',0)
    
    
    filepath_m5 = './sequences/phylogeny_Alpham300_300_K40_M5_100real/2022_12_07_10_34_11_Sectors_nspins200_flips3000_seed_76_deltanormal_M5_Kappa40_G11_realnbr0.h5'
    filepath_m15 = './sequences/phylogeny_Alpham300_300_K40_M15_100real/2022_12_07_10_35_05_Sectors_nspins200_flips3000_seed_76_deltanormal_M15_Kappa40_G11_realnbr0.h5'
    filepath_eq = './sequences/EQ_K40_alpham300_300/2022_12_06_18_54_13_Sectors_nspins200_flips3000_nseq2048_seed_76_deltanormal_Alphamin-300_Alphamax300_K40_realnbr0.h5'
    pseudocount = 0.00001
    eigvalC_s,eigvalICOD_s,pval1_s,pval2_s = Eigenvalues(filepath_eq,'Alpha2',90,'Kappa',pseudocount)
    eigvalC_m15_s,eigvalICOD_m15_s,pval1_s,pval2_s = Eigenvalues(filepath_m15,'Alpha',90,'Kappa',pseudocount)
    eigvalC_m5_s,eigvalICOD_m5_s,pval1_s,pval2_s = Eigenvalues(filepath_m5,'Alpha',90,'Kappa',pseudocount)
   
    sca_m5_s,_,pval1,pval2 = EigenvaluesEigenvectorsSCA(filepath_m5,'Alpha',90,'Kappa',0)
    sca_m15_s,_,pval1,pval2 = EigenvaluesEigenvectorsSCA(filepath_m15,'Alpha',90,'Kappa',0)
    sca_eq_s,_,pval1,pval2 = EigenvaluesEigenvectorsSCA(filepath_eq,'Alpha2',90,'Kappa',0)
    
    
    
    pval1 = int(pval1[0])
    lwdth = 2
    
    fig, ax1 = plt.subplots(1,3,figsize = (16,7.2))

    ax1[1].set_title('Covariance',fontweight="bold")

    nbrsites= eigvalICOD.shape[0]
    rank = np.arange(nbrsites)
    
    ax1[1].plot(rank,eigvalC,'-o',color = newcolorspal_shades[0],linewidth = lwdth)
    ax1[1].plot(rank,eigvalC_s,'-^', color = newcolorspal[0],linewidth = lwdth)

    ax1[1].plot(rank,eigvalC_m15,'-o',color = newcolorspal_shades[1],linewidth = lwdth)
    ax1[1].plot(rank,eigvalC_m15_s,'-^', color = newcolorspal[1],linewidth = lwdth)
    
    ax1[1].plot(rank,eigvalC_m5,'-o',color = newcolorspal_shades[2],linewidth = lwdth)    
    ax1[1].plot(rank,eigvalC_m5_s,'-^', color = newcolorspal[2],linewidth = lwdth)

    
    ax1[0].set_ylabel('Eigenvalue')
    ax1[1].set_xlabel('Rank')
    
    ax1[1].set_yscale('log')
    
    ax1[1].set_xlim([-4,202])
    ax1[1].set_ylim([0.030023161974780443,10])

    left, bottom, width, height = [0.41, 0.7, 0.1, 0.18]
    ax2 = fig.add_axes([left, bottom, width, height])
    cutoff_xscale = 10
    ax2.plot(rank[:cutoff_xscale],eigvalC[:cutoff_xscale],'--o',color = newcolorspal_shades[0],linewidth = lwdth)
    ax2.plot(rank[:cutoff_xscale],eigvalC_s[:cutoff_xscale],'-^', color = newcolorspal[0],linewidth = lwdth)
    
    ax2.plot(rank[:cutoff_xscale],eigvalC_m15[:cutoff_xscale],'-o',color = newcolorspal_shades[1],linewidth = lwdth)
    ax2.plot(rank[:cutoff_xscale],eigvalC_m15_s[:cutoff_xscale],'-^', color = newcolorspal[1],linewidth = lwdth)
   
    ax2.plot(rank[:cutoff_xscale],eigvalC_m5[:cutoff_xscale],'-o',color = newcolorspal_shades[2],linewidth = lwdth)
    ax2.plot(rank[:cutoff_xscale],eigvalC_m5_s[:cutoff_xscale],'-^', color = newcolorspal[2],linewidth = lwdth)

    ax2.set_xlim([-0.2,cutoff_xscale-0.8])
    

    ax1[0].set_title('ICOD',fontweight="bold")

    nbrsites= sca_m50.shape[0]
    rank = np.arange(nbrsites)
    
    ax1[0].plot(rank,eigvalICOD,'-o', color = newcolorspal_shades[0],linewidth = lwdth)
    ax1[0].plot(rank,eigvalICOD_s,'-^',color = newcolorspal[0],linewidth = lwdth)

    ax1[0].plot(rank,eigvalICOD_m15,'-o', color = newcolorspal_shades[1],linewidth = lwdth)
    ax1[0].plot(rank,eigvalICOD_m15_s,'-^',color = newcolorspal[1],linewidth = lwdth)
    
    ax1[0].plot(rank,eigvalICOD_m5,'-o', color = newcolorspal_shades[2],linewidth = lwdth)
    ax1[0].plot(rank,eigvalICOD_m5_s,'-^',color = newcolorspal[2],linewidth = lwdth)
    
    custom_lines = ["No selection:",
                    Line2D([0], [0], color=newcolorspal_shades[0], lw=lwdth,linestyle = '-',marker = 'o'),
                    Line2D([0], [0], color=newcolorspal_shades[1], lw=lwdth,linestyle = '-',marker = 'o'),
                    Line2D([0], [0], color=newcolorspal_shades[2], lw=lwdth,linestyle = '-',marker = 'o'),
                    '',
                    "Selection:",
                    Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '-',marker = '^'),
                    Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '-',marker = '^'),
                    Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '-',marker = '^')]

    
    labels = ['','$\mu = 50$','$\mu = 15$','$\mu = 5$','','','No phylogeny','$\mu = 15$','$\mu = 5$']
    

    ax1[0].set_xlabel('Rank')
    ax1[0].set_xlim([-4,202])
    ax1[0].set_ylim([-4.8,12])
    
    left, bottom, width, height = [0.15, 0.7, 0.1, 0.18]
    ax3 = fig.add_axes([left, bottom, width, height])
    cutoff_xscale = 5
    ax3.plot(rank[:cutoff_xscale],eigvalICOD[:cutoff_xscale],'-o',color = newcolorspal_shades[0],linewidth = lwdth)
    ax3.plot(rank[:cutoff_xscale],eigvalICOD_s[:cutoff_xscale],'-^', color = newcolorspal[0],linewidth = lwdth)
    
    ax3.plot(rank[:cutoff_xscale],eigvalICOD_m15[:cutoff_xscale],'-o',color = newcolorspal_shades[1],linewidth = lwdth)
    ax3.plot(rank[:cutoff_xscale],eigvalICOD_m15_s[:cutoff_xscale],'-^', color = newcolorspal[1],linewidth = lwdth)
   
    ax3.plot(rank[:cutoff_xscale],eigvalICOD_m5[:cutoff_xscale],'-o',color = newcolorspal_shades[2],linewidth = lwdth)
    ax3.plot(rank[:cutoff_xscale],eigvalICOD_m5_s[:cutoff_xscale],'-^', color = newcolorspal[2],linewidth = lwdth)

    ax3.set_xlim([-0.2,cutoff_xscale-0.8])
    ax3.set_xticks([0,3])


    ax1[2].set_title('SCA',fontweight="bold")

    nbrsites= eigvalICOD.shape[0]
    rank = np.arange(nbrsites)
    
    ax1[2].plot(rank,sca_m50,'-o', color = newcolorspal_shades[0],linewidth = lwdth)
    ax1[2].plot(rank,sca_eq_s,'-^',color = newcolorspal[0],linewidth = lwdth)

    ax1[2].plot(rank,sca_m15,'-o', color = newcolorspal_shades[1],linewidth = lwdth)
    ax1[2].plot(rank,sca_m15_s,'-^',color = newcolorspal[1],linewidth = lwdth)
    
    ax1[2].plot(rank,sca_m5,'-o', color = newcolorspal_shades[2],linewidth = lwdth)
    ax1[2].plot(rank,sca_m5_s,'-^',color = newcolorspal[2],linewidth = lwdth)
    
    left, bottom, width, height = [0.65, 0.7, 0.12, 0.18]
    ax4 = fig.add_axes([left, bottom, width, height])
    cutoff_xscale = 25

    
    ax4.plot(rank[:cutoff_xscale],sca_m15[:cutoff_xscale],'-o',color = newcolorspal_shades[1],linewidth = lwdth)
    ax4.plot(rank[:cutoff_xscale],sca_m15_s[:cutoff_xscale],'-^', color = newcolorspal[1],linewidth = lwdth)
   
    ax4.plot(rank[:cutoff_xscale],sca_m5[:cutoff_xscale],'-o',color = newcolorspal_shades[2],linewidth = lwdth)
    ax4.plot(rank[:cutoff_xscale],sca_m5_s[:cutoff_xscale],'-^', color = newcolorspal[2],linewidth = lwdth)
    ax4.plot(rank[:cutoff_xscale],sca_m50[:cutoff_xscale],'-o',color = newcolorspal_shades[0],linewidth = lwdth)
    ax4.plot(rank[:cutoff_xscale],sca_eq_s[:cutoff_xscale],'-^', color = newcolorspal[0],linewidth = lwdth)
    ax4.set_xlim([-0.8,cutoff_xscale-0.2])
    ax4.set_ylim([-0.07,1])
    ax4.set_xticks([0,10,20])

    
    custom_lines = ["No selection:",
                    Line2D([0], [0], color=newcolorspal_shades[0], lw=lwdth,linestyle = '-',marker = 'o'),
                    Line2D([0], [0], color=newcolorspal_shades[1], lw=lwdth,linestyle = '-',marker = 'o'),
                    Line2D([0], [0], color=newcolorspal_shades[2], lw=lwdth,linestyle = '-',marker = 'o'),
                    '',
                    "Selection:",
                    Line2D([0], [0], color=newcolorspal[0], lw=lwdth,linestyle = '-',marker = '^'),
                    Line2D([0], [0], color=newcolorspal[1], lw=lwdth,linestyle = '-',marker = '^'),
                    Line2D([0], [0], color=newcolorspal[2], lw=lwdth,linestyle = '-',marker = '^')]

    
    labels = ['','$\mu = 50$','$\mu = 15$','$\mu = 5$','','','No phylogeny','$\mu = 15$','$\mu = 5$']
    
    ax1[2].legend(custom_lines,labels,bbox_to_anchor = (1, 1.05), loc='upper left')

    ax1[2].set_xlabel('Rank')
    ax1[2].set_xlim([-4,202])
    ax1[2].set_ylim([-0.1,3.4])


    fig.subplots_adjust(wspace=0.335,hspace = 0.2,top= 0.93, bottom = 0.13, left = 0.075,right = 0.79)

    
def RecoveryFullspectrum(ev,delta):
    nbrank,_= ev.shape
    rec = np.zeros(nbrank)
    for i in range(0,nbrank):
        rec[i] = RecoveryOneVector(ev[:,i],delta)
    return rec
    
def PlotREC_FULLSPECTRUM_subplots():
    
    filepath_m5 = './sequences/phylogeny_Alpham300_300_K40_M5_100real/2022_12_07_10_34_11_Sectors_nspins200_flips3000_seed_76_deltanormal_M5_Kappa40_G11_realnbr0.h5'
    filepath_m15 = './sequences/phylogeny_Alpham300_300_K40_M15_100real/2022_12_07_10_35_05_Sectors_nspins200_flips3000_seed_76_deltanormal_M15_Kappa40_G11_realnbr0.h5'
    filepath_eq = './sequences/EQ_K40_alpham300_300/2022_12_06_18_54_13_Sectors_nspins200_flips3000_nseq2048_seed_76_deltanormal_Alphamin-300_Alphamax300_K40_realnbr0.h5'
    pseudocount = 0.00001
    file = h5py.File(filepath_eq,'r')
    delta = np.array(file['Delta'])
    file.close()
    
    random_expectation = RandomExpectation('./sequences/EQ_K40_alpham300_300/')
    eigvectC,eigvectICOD,pval1,pval2_s = Eigenvectors(filepath_eq,'Alpha2',90,'Kappa',pseudocount)
    eigvectC_m15,eigvectICOD_m15,pval1,pval2 = Eigenvectors(filepath_m15,'Alpha',90,'Kappa',pseudocount)
    eigvectC_m5,eigvectICOD_m5,pval1,pval2 = Eigenvectors(filepath_m5,'Alpha',90,'Kappa',pseudocount)
   
    sca_m5,ev_scam5,pval1,pval2 = EigenvaluesEigenvectorsSCA(filepath_m5,'Alpha',90,'Kappa',0)
    sca_m15,ev_scam15,pval1,pval2 = EigenvaluesEigenvectorsSCA(filepath_m15,'Alpha',90,'Kappa',0)
    sca_eq,ev_scaeq,pval1,pval2 = EigenvaluesEigenvectorsSCA(filepath_eq,'Alpha2',90,'Kappa',0)
    
    rec_icod_eq = RecoveryFullspectrum(eigvectICOD, delta)
    rec_icod_m15 = RecoveryFullspectrum(eigvectICOD_m15, delta)
    rec_icod_m5 = RecoveryFullspectrum(eigvectICOD_m5, delta)
    
    rec_c_eq = RecoveryFullspectrum(eigvectC, delta)
    rec_c_m15 = RecoveryFullspectrum(eigvectC_m15, delta)
    rec_c_m5 = RecoveryFullspectrum(eigvectC_m5, delta)
    
    rec_sca_eq = RecoveryFullspectrum(ev_scaeq, delta)
    rec_sca_m15 = RecoveryFullspectrum(ev_scam15, delta)
    rec_sca_m5 = RecoveryFullspectrum(ev_scam5, delta)
    
    pval1 = int(pval1[0])
    lwdth = 2
    
    
    idxc_eq = 0
    idxc_m15 = 1
    idxc_m5 = 2
    
    fig, ax1 = plt.subplots(3,3,figsize = (16,7.2))

    ax1[0,1].set_title('Covariance',fontweight="bold")

    nbrsites= eigvectICOD.shape[0]
    rank = np.arange(nbrsites)
    
    ax1[0,1].bar(rank,rec_c_eq,color = newcolorspal[idxc_eq])
    ax1[1,1].bar(rank,rec_c_m15,color = newcolorspal[idxc_m15])
    ax1[2,1].bar(rank,rec_c_m5,color = newcolorspal[idxc_m5])
    
    ax1[0,1].hlines(random_expectation,rank[0],rank[-1],linestyle = '--',color = 'k')
    ax1[1,1].hlines(random_expectation,rank[0],rank[-1],linestyle = '--',color = 'k')
    ax1[2,1].hlines(random_expectation,rank[0],rank[-1],linestyle = '--',color = 'k')
    
    ax1[2,1].set_xlabel('Rank')
    
    
    
    
    ax1[0,0].set_title('ICOD',fontweight="bold")

    nbrsites= eigvectICOD.shape[0]
    rank = np.arange(nbrsites)
    
    ax1[0,0].bar(rank,rec_icod_eq,color = newcolorspal[idxc_eq])
    ax1[1,0].bar(rank,rec_icod_m15,color = newcolorspal[idxc_m15])
    ax1[2,0].bar(rank,rec_icod_m5,color = newcolorspal[idxc_m5])

    ax1[0,0].hlines(random_expectation,rank[0],rank[-1],linestyle = '--',color = 'k')
    ax1[1,0].hlines(random_expectation,rank[0],rank[-1],linestyle = '--',color = 'k')
    ax1[2,0].hlines(random_expectation,rank[0],rank[-1],linestyle = '--',color = 'k')
    
    
    ax1[2,0].set_xlabel('Rank')
    
    ax1[0,0].set_ylabel('Recovery')
    ax1[1,0].set_ylabel('Recovery')
    ax1[2,0].set_ylabel('Recovery')



    ax1[0,2].set_title('SCA',fontweight="bold")

    ax1[0,2].bar(rank,rec_sca_eq,color = newcolorspal[idxc_eq])
    ax1[1,2].bar(rank,rec_sca_m15,color = newcolorspal[idxc_m15])
    ax1[2,2].bar(rank,rec_sca_m5,color = newcolorspal[idxc_m5])
    
    ax1[0,2].hlines(random_expectation,rank[0],rank[-1],linestyle = '--',color = 'k')
    ax1[1,2].hlines(random_expectation,rank[0],rank[-1],linestyle = '--',color = 'k')
    ax1[2,2].hlines(random_expectation,rank[0],rank[-1],linestyle = '--',color = 'k')
    
    ax1[2,2].set_xlabel('Rank')
    

    
    for i in range(0,3):
        for j in range(0,3):
            ax1[i,j].set_ylim([0,1])
            ax1[i,j].set_xlim([-2,201])
    # fig.subplots_adjust(wspace=0.335,hspace = 0.2,top= 0.93, bottom = 0.13, left = 0.075,right = 0.79)
    ax1[0, 0].text(-0.35, -0.1, r'No phylogeny', transform=ax1[0, 0].transAxes, size=fontsize_test,rotation = 'vertical',weight = 'bold')
    ax1[1, 0].text(-0.35, 0.2, r'$\mathbf{\mu = 15}$', transform=ax1[1, 0].transAxes, size=fontsize_test,rotation = 'vertical',weight = 'bold')
    ax1[2, 0].text(-0.35, 0.25, r'$\mathbf{\mu = 5}$', transform=ax1[2, 0].transAxes, size=fontsize_test,rotation = 'vertical',weight = 'bold')
    fig.subplots_adjust(wspace=0.2,hspace = 0.35,top= 0.945, bottom = 0.115, left = 0.095,right = 0.98)


def NewScoreEarliestGenMutSingleSite(allchains,number_generations):
    
    newscore_em = []
    nbrseq,nbrpos = allchains.shape
    for s1 in np.arange(nbrpos):

        cl1 = allchains[:,s1]
        init_states1 = cl1[0]

        stop = False
        
        for k in range(0,number_generations+1):
            for s in range(0,pow(2,k)):
                idx = pow(2,k)-1+s
                if cl1[idx] != cl1[0]:
                        newscore_em.append(k)
                        stop = True
                        break
            if stop:
                break
        if not stop:
            newscore_em.append(number_generations + 1)
            
    return newscore_em

def ComputeEVScoresOnereal(path,pseudocount):
    ev_cov,ev_icod,ev_sca,allchains,nbrgenerations = EigenvectorsSavePhylogeny(path,pseudocount)
    _,nbrpos = allchains.shape
    listpairs = []
    for j in range(0,nbrpos):
        for i in range(j+1,nbrpos):
            listpairs.append([j,i])

    gscore_singlesite = NewScoreEarliestGenMutSingleSite(allchains,nbrgenerations)
    
    return ev_cov,ev_icod,ev_sca,gscore_singlesite

def PrepareDataViolin(list_icod_values,list_cov_values,list_sca_values,list_gvalues):
    list_cov_values = np.array(list_cov_values)
    list_icod_values = np.array(list_icod_values)
    list_sca_values = np.array(list_sca_values)
    list_gvalues = np.array(list_gvalues)
    pos = np.unique(list_gvalues)
    
    cov = [list_cov_values[list_gvalues == p] for p in pos]
    icod = [list_icod_values[list_gvalues == p] for p in pos]
    sca = [list_sca_values[list_gvalues == p] for p in pos]
    
    return cov,icod,sca,pos

def Scores_Spectrum(filepath,pseudocount,nbrsitessect,idxcov,idxicod):
    listfiles = os.listdir(filepath)

    list_cov_values_sectors = []
    list_icod_values_sectors = []
    list_sca_values_sectors = []
    
    list_cov_values_nonsectors = []
    list_icod_values_nonsectors = []
    list_sca_values_nonsectors = []
    
    list_gvalues_sector = []
        
    list_gvalues_nonsector = []
    

    
    for f in tqdm(listfiles):
        if not(f.startswith('.')):
            path_tmp = filepath+'/'+f
            ev_cov,ev_icod,ev_sca,gsinglsite = ComputeEVScoresOnereal(path_tmp,pseudocount)
            ev_cov= np.abs(ev_cov)
            ev_icod = np.abs(ev_icod)
            ev_sca = np.abs(ev_sca)
            list_gvalues_sector = list_gvalues_sector + gsinglsite[:nbrsitessect]
            list_gvalues_nonsector = list_gvalues_nonsector + gsinglsite[nbrsitessect:]
            
            
            list_cov_values_sectors = list_cov_values_sectors + list(ev_cov[:nbrsitessect,idxcov])
            list_icod_values_sectors = list_icod_values_sectors + list(ev_icod[:nbrsitessect,idxicod])
            list_sca_values_sectors = list_sca_values_sectors + list(ev_sca[:nbrsitessect,idxicod])

            list_cov_values_nonsectors = list_cov_values_nonsectors + list(ev_cov[nbrsitessect:,idxcov])
            list_icod_values_nonsectors = list_icod_values_nonsectors + list(ev_icod[nbrsitessect:,idxicod])
            list_sca_values_nonsectors = list_sca_values_nonsectors + list(ev_sca[nbrsitessect:,idxicod])

    # import pdb;pdb.set_trace()
    
    cov_sector,icod_sector,sca_sector,pos_sect = PrepareDataViolin(list_icod_values_sectors,list_cov_values_sectors,list_sca_values_sectors,list_gvalues_sector)
    cov_nonsector,icod_nonsector,sca_nonsector,pos_nonsect = PrepareDataViolin(list_icod_values_nonsectors,list_cov_values_nonsectors,list_sca_values_nonsectors,list_gvalues_nonsector)
    
    return cov_sector,icod_sector,sca_sector,pos_sect,cov_nonsector,icod_nonsector,sca_nonsector,pos_nonsect


def CorrelationGscoresEigenvectors_final():
    mu = 50 
    pseudocount = 0.00001

    nbrsitessect = 20

    idxcov = -1
    idxicod = 0
    ## the order of the ev is flipped -> big eigenvalue is index 0 and small -1
    # cov_sector_m1,icod_sector_0,sca_sector_0,pos_sect,cov_nonsector_m1,icod_nonsector_0,sca_nonsector_0,pos_nonsect = Scores_Spectrum(filepath,pseudocount,nbrsitessect,-1,0)
    
    # cov_sector_m1_m5,icod_sector_0_m5,sca_sector_0_m5,pos_sect_m5,cov_nonsector_m1_m5,icod_nonsector_0_m5,sca_nonsector_0_m5,pos_nonsect_m5 = Scores_Spectrum(filepathm5,pseudocount,nbrsitessect,-1,0)
    

    ptheq = './results/gscores/EQ/'
    cov_sector_m1 = np.load(ptheq+'Covariance/cov_sector_m1.npy',allow_pickle=True)
    # pdb.set_trace()
    cov_nonsector_m1 = np.load(ptheq+'Covariance/cov_nonsector_m1.npy',allow_pickle=True)
    
    icod_sector_0 = np.load(ptheq+'ICOD/icod_sector_0.npy',allow_pickle=True)
    icod_nonsector_0 = np.load(ptheq+'ICOD/icod_nonsector_0.npy',allow_pickle=True)
    
    sca_sector_0 = np.load(ptheq+'SCA/sca_sector_0.npy',allow_pickle=True)
    sca_nonsector_0 = np.load(ptheq+'SCA/sca_nonsector_0.npy',allow_pickle=True)
    
    pos_sect = np.load(ptheq+'pos_sect.npy')
    pos_nonsect = np.load(ptheq+'pos_nonsect.npy')
    
    pthm5 = './results/gscores/m5/'
    cov_sector_m1_m5 = np.load(pthm5+'Covariance/cov_sector_m1_m5.npy',allow_pickle=True)
    cov_nonsector_m1_m5 = np.load(pthm5+'Covariance/cov_nonsector_m1_m5.npy',allow_pickle=True)
    
    icod_sector_0_m5 = np.load(pthm5+'ICOD/icod_sector_0_m5.npy',allow_pickle=True)
    icod_nonsector_0_m5 = np.load(pthm5+'ICOD/icod_nonsector_0_m5.npy',allow_pickle=True)
    
    sca_sector_0_m5 = np.load(pthm5+'SCA/sca_sector_0_m5.npy',allow_pickle=True)
    sca_nonsector_0_m5 = np.load(pthm5+'SCA/sca_nonsector_0_m5.npy',allow_pickle=True)

    pos_sect_m5 = np.load(pthm5+'pos_sect_m5.npy')
    pos_nonsect_m5 = np.load(pthm5+'pos_nonsect_m5.npy')
    
    # pdb.set_trace()

    fig,axs = plt.subplots(2,3,figsize = (16,7.21))
    p = axs[0,1].violinplot(cov_sector_m1,pos_sect,showextrema = False)
    plots = axs[0,1].violinplot(cov_nonsector_m1,pos_nonsect,showextrema = False)

    for pc in plots['bodies']:
        pc.set_facecolor('C3')
        pc.set_alpha(0.6)
    for pc in p['bodies']:
        pc.set_alpha(0.6)
    p=axs[1,1].violinplot(cov_sector_m1_m5,pos_sect_m5,showextrema = False)
    plots =axs[1,1].violinplot(cov_nonsector_m1_m5,pos_nonsect_m5,showextrema = False)
    for pc in plots['bodies']:
        pc.set_facecolor('C3')
        pc.set_alpha(0.6)
    for pc in p['bodies']:
        pc.set_alpha(0.6)
    p=axs[0,0].violinplot(icod_sector_0,pos_sect,showextrema = False)
    plots =axs[0,0].violinplot(icod_nonsector_0,pos_nonsect,showextrema = False)
    for pc in plots['bodies']:
        pc.set_facecolor('C3')
        pc.set_alpha(0.6)
    for pc in p['bodies']:
        pc.set_alpha(0.6)
    p=axs[1,0].violinplot(icod_sector_0_m5,pos_sect_m5,showextrema = False)
    plots =axs[1,0].violinplot(icod_nonsector_0_m5,pos_nonsect_m5,showextrema = False)
    for pc in plots['bodies']:
        pc.set_facecolor('C3')
        pc.set_alpha(0.6)
    for pc in p['bodies']:
        pc.set_alpha(0.6)
        
    p=axs[0,2].violinplot(sca_sector_0,pos_sect,showextrema = False)
    plots =axs[0,2].violinplot(sca_nonsector_0,pos_nonsect,showextrema = False)
    for pc in plots['bodies']:
        pc.set_facecolor('C3')
        pc.set_alpha(0.6)
    for pc in p['bodies']:
        pc.set_alpha(0.6)
    p=axs[1,2].violinplot(sca_sector_0_m5,pos_sect_m5,showextrema = False)
    plots =axs[1,2].violinplot(sca_nonsector_0_m5,pos_nonsect_m5,showextrema = False)
    for pc in plots['bodies']:
        pc.set_facecolor('C3')
        pc.set_alpha(0.6)
    for pc in p['bodies']:
        pc.set_alpha(0.6)
    custom_lines = [Line2D([0], [0], color='C0',lw=4,linestyle = '-',alpha = 0.6),
                    Line2D([0], [0], color='C3',lw=4,linestyle = '-',alpha = 0.6)]
    labels = ['Sector','Non sector']

    legend1 = axs[0,2].legend(custom_lines,labels, bbox_to_anchor = (1.01,1),loc='upper left')
    
    max_lim = np.max(pos_sect)
    axs[0,0].set_xticks(np.arange(max_lim)+1)
    axs[0,1].set_xticks(np.arange(max_lim)+1)
    axs[0,2].set_xticks(np.arange(max_lim)+1)
    max_lim = max(np.max(pos_sect),np.max(pos_sect_m5))
    axs[1,0].set_xticks(np.arange(max_lim)+1)
    axs[1,1].set_xticks(np.arange(max_lim)+1)
    axs[1,2].set_xticks(np.arange(max_lim)+1)

    
    axs[1,0].set_ylabel('Eigenvector \ncomponent')
    axs[0,0].set_ylabel('Eigenvector \ncomponent')

    axs[0,1].set_title('C $\lambda_{min}$', weight = 'bold')
    axs[0,0].set_title('ICOD $\Lambda_{max}$', weight = 'bold')
    axs[0,2].set_title('SCA $\lambda_{max}$', weight = 'bold')

    axs[0, 0].text(-0.55, 0.35, r'$\mathbf{\mu = 50}$', transform=axs[0, 0].transAxes, size=fontsize_test,rotation = 'vertical',weight = 'bold')
    axs[1, 0].text(-0.55, 0.35, r'$\mathbf{\mu = 5}$', transform=axs[1, 0].transAxes, size=fontsize_test,rotation = 'vertical',weight = 'bold')
    
    axs[1, 1].text(-0.15, -0.35, 'Earliest mutation generation G', transform=axs[1, 1].transAxes, size=fontsize_test)

    fig.subplots_adjust(wspace=0.3,hspace = 0.255,top= 0.935, bottom = 0.155, left = 0.115,right = 0.815)

    
def CorrelationGscoresCovariance_final():
    mu = 5
    filepath = './sequences/phylogeny_Alpha90_K40_M'+str(mu)+'_savephylogeny'
    pseudocount = 0.00001
    listfiles = os.listdir(filepath)
    nbrsitessect = 20

    idxcov = -1
    idxicod = 0
    ## the order of the ev is flipped -> big eigenvalue is index 0 and small -1
    cov_sector_m1,_,_,pos_sectm1,cov_nonsector_m1,_,_,pos_nonsectm1 = Scores_Spectrum(filepath,pseudocount,nbrsitessect,-1,0)
    cov_sector_0,_,_,pos_sect0,cov_nonsector_0,_,_,pos_nonsect0 = Scores_Spectrum(filepath,pseudocount,nbrsitessect,0,0)

    fig,axs = plt.subplots(2,1,figsize = (11,7.2))
    p = axs[1].violinplot(cov_sector_m1,pos_sectm1,showextrema = False)
    plots = axs[1].violinplot(cov_nonsector_m1,pos_nonsectm1,showextrema = False)

    for pc in plots['bodies']:
        pc.set_facecolor('C3')
        pc.set_alpha(0.6)
    for pc in p['bodies']:
        pc.set_alpha(0.6)

    p=axs[0].violinplot(cov_sector_0,pos_sect0,showextrema = False)
    plots =axs[0].violinplot(cov_nonsector_0,pos_nonsect0,showextrema = False)
    for pc in plots['bodies']:
        pc.set_facecolor('C3')
        pc.set_alpha(0.6)
    for pc in p['bodies']:
        pc.set_alpha(0.6)

    custom_lines = [Line2D([0], [0], color='C0',lw=4,linestyle = '-',alpha = 0.6),
                    Line2D([0], [0], color='C3',lw=4,linestyle = '-',alpha = 0.6)]
    labels = ['Sector','Non sector']

    legend1 = axs[0].legend(custom_lines,labels, bbox_to_anchor = (1.01,1),loc='upper left')
    axs[0].set_xticks([])

    max_lim = max(np.max(pos_sect0),np.max(pos_sectm1))
    axs[1].set_xticks(np.arange(max_lim)+1)


    axs[1].set_xlabel('Earliest mutation generation G')

    axs[1].text(-0.2, 0.5, 'Eigenvector component', transform=axs[1].transAxes, size=fontsize_test,rotation = 'vertical')
  
    
    axs[0].text(0.7, 0.7, r'$\lambda_{max}$', transform=axs[0].transAxes, size=fontsize_test)
    axs[1].text(0.7, 0.7, r'$\lambda_{min}$', transform=axs[1].transAxes, size=fontsize_test)
    axs[1].set_yticks([0,0.4,0.8])

    
    fig.subplots_adjust(wspace=0.2,hspace = 0,top= 0.948, bottom = 0.148, left = 0.128,right = 0.698)

def Block_Data(sequences):
    covariance_matrix = ComputeCorrelationMatrix2(sequences, 0.00001)
    covariance_matrix_0pseudocount = ComputeCorrelationMatrix2(sequences, 0)
    icod = np.linalg.inv(covariance_matrix_0pseudocount)
    np.fill_diagonal(icod,0)

    
    approx_block = icod.copy()
    approx_block[:20,20:] = 0
    approx_block[20:,:20] = 0

    eigenvalues_icod, eigenvectors_icod = np.linalg.eigh(icod)
    eigenvalues_block, eigenvectors_block = np.linalg.eigh(approx_block)
    
    # import pdb;pdb.set_trace()
    
    blockA = approx_block[:20,:20]
    blockB = approx_block[20:,20:]
    
    eigenvalues_A, _ = np.linalg.eigh(blockA)
    eigenvalues_B,_ = np.linalg.eigh(blockB)
    
    egAB = np.concatenate((eigenvalues_A,eigenvalues_B))
    
    idxs_ab = np.arange(egAB.shape[0])
    idxs = np.argsort(egAB)
    # idxs_ab = idxs_ab[idxs]
    # pdb.set_trace()
    
    egAB_sorted = egAB[idxs]
    rk_sect = []
    rk_nonsect = []
    colors = ['C2']*20 + ['C3']*180
    # for i in range(0,egAB.shape[0]):
    col = np.array(colors)
    col_ord = col[idxs]
    
    return np.flip(eigenvalues_icod),np.flip(egAB_sorted),np.flip(col_ord),icod,approx_block


def Block_DataInverseC(sequences):
    covariance_matrix = ComputeCorrelationMatrix2(sequences, 0.00001)
    covariance_matrix_0pseudocount = ComputeCorrelationMatrix2(sequences, 0)
    icod = np.linalg.inv(covariance_matrix_0pseudocount)

    approx_block = icod.copy()
    approx_block[:20,20:] = 0
    approx_block[20:,:20] = 0

    eigenvalues_icod, eigenvectors_icod = np.linalg.eigh(icod)
    eigenvalues_block, eigenvectors_block = np.linalg.eigh(approx_block)
    

    blockA = approx_block[:20,:20]
    blockB = approx_block[20:,20:]
    
    eigenvalues_A, _ = np.linalg.eigh(blockA)
    eigenvalues_B,_ = np.linalg.eigh(blockB)
    
    egAB = np.concatenate((eigenvalues_A,eigenvalues_B))
    
    idxs_ab = np.arange(egAB.shape[0])
    idxs = np.argsort(egAB)

    
    egAB_sorted = egAB[idxs]
    rk_sect = []
    rk_nonsect = []
    colors = ['C2']*20 + ['C3']*180

    col = np.array(colors)
    col_ord = col[idxs]
    
    return np.flip(eigenvalues_icod),np.flip(egAB_sorted),np.flip(col_ord),icod,approx_block


def Block_theoretical(delta,kappa):
    kappa = kappa/np.dot(delta,delta)

    icod = kappa*np.tensordot(delta,delta,axes =0)

    np.fill_diagonal(icod,0)

    approx_block = icod.copy()
    approx_block[:20,20:] = 0
    approx_block[20:,:20] = 0

    eigenvalues_icod, eigenvectors_icod = np.linalg.eigh(icod)
    eigenvalues_block, eigenvectors_block = np.linalg.eigh(approx_block)

    
    blockA = approx_block[:20,:20]
    blockB = approx_block[20:,20:]
    
    eigenvalues_A,_ = np.linalg.eigh(blockA)
    eigenvalues_B,_ = np.linalg.eigh(blockB)
    
    egAB = np.concatenate((eigenvalues_A,eigenvalues_B))
    
    idxs_ab = np.arange(egAB.shape[0])
    idxs = np.argsort(egAB)
    # idxs_ab = idxs_ab[idxs]
    # pdb.set_trace()
    
    egAB_sorted = egAB[idxs]
    rk_sect = []
    rk_nonsect = []
    
    newcolorspal
    
    colors = ['C2']*20 + ['C3']*180
    # for i in range(0,egAB.shape[0]):
    col = np.array(colors)
    col_ord = col[idxs]
    

    return np.flip(eigenvalues_icod), np.flip(egAB_sorted), np.flip(col_ord)


    
def BlockMatrixFinal_MAINversion():
    ptheq = './sequences/EQ_K40_alpham300_300_100real/'
    
    filepath = ptheq+'2023_04_24_11_47_24_Sectors_nspins200_flips3000_nseq2048_seed_34_deltanormal_Alphamin-300_Alphamax300_K40_realnbr0.h5'
    fpath2 =  "./sequences/EQ_K40_alpham300_300_100real/2023_04_24_11_47_24_Sectors_nspins200_flips3000_nseq2048_seed_34_deltanormal_Alphamin-300_Alphamax300_K40_realnbr1.h5"
    fpath3 =  "./sequences/EQ_K40_alpham300_300_100real/2023_04_24_11_47_24_Sectors_nspins200_flips3000_nseq2048_seed_34_deltanormal_Alphamin-300_Alphamax300_K40_realnbr2.h5"
    fpath4 =  "./sequences/EQ_K40_alpham300_300_100real/2023_04_24_11_47_24_Sectors_nspins200_flips3000_nseq2048_seed_34_deltanormal_Alphamin-300_Alphamax300_K40_realnbr3.h5"
    fpath5 =  "./sequences/EQ_K40_alpham300_300_100real/2023_04_24_11_47_24_Sectors_nspins200_flips3000_nseq2048_seed_34_deltanormal_Alphamin-300_Alphamax300_K40_realnbr4.h5"
    fpath6 =  "./sequences/EQ_K40_alpham300_300_100real/2023_04_24_11_47_24_Sectors_nspins200_flips3000_nseq2048_seed_34_deltanormal_Alphamin-300_Alphamax300_K40_realnbr5.h5"
    fpath7 =  "./sequences/EQ_K40_alpham300_300_100real/2023_04_24_11_47_24_Sectors_nspins200_flips3000_nseq2048_seed_34_deltanormal_Alphamin-300_Alphamax300_K40_realnbr6.h5"

    sequences,delta1,param1,val =  OpenFile(filepath,'Alpha','Kappa')
    sequences3,delta3,param1,val =  OpenFile(fpath2,'Alpha','Kappa')
    sequences4,delta4,param1,val =  OpenFile(fpath3,'Alpha','Kappa')
    sequences5,delta5,param1,val =  OpenFile(fpath4,'Alpha','Kappa')
    sequences6,delta6,param1,val =  OpenFile(fpath5,'Alpha','Kappa')
    sequences7,delta7,param1,val =  OpenFile(fpath6,'Alpha','Kappa')
    sequences8,delta8,param1,val =  OpenFile(fpath7,'Alpha','Kappa')
    

    seq = np.concatenate((sequences,sequences3,sequences4,sequences5,sequences6,sequences7,sequences8 ),axis =1)
    file = h5py.File(filepath,'r')
    delta = np.array(file['Delta'])
    file.close()
    valparamt = 0
    chains_nbig = seq[param1 == valparamt,:14000,:][0]
    chains_nsmall = sequences[param1 == valparamt,:2000,:][0]


    eigenvalues_icod_dataNsmall, egAB_sorted_dataNsmall, col_ord_dataNsmall,icod_nsmall,approxblock_nsmall = Block_Data(chains_nsmall)
    
    eigenvalues_icod_dataNbig, egAB_sorted_dataNbig, col_ord_dataNbig,icod,approxblock = Block_Data(chains_nbig)
    
    eigenvalues_icod_th, egAB_sorted_th, col_ord_th = Block_theoretical(delta/2,val/4)

    eigenvalues_ic_dataNbig, egAB_ic_sorted_dataNbig, col_ic_ord_dataNbig,ic,approxblock_Nbig = Block_DataInverseC(chains_nbig)
    
    eigenvalues_ic_dataNsmall, egAB_ic_sorted_dataNsmall, col_ic_ord_dataNsmall,ic_nsmall,approxblock_nsmall = Block_DataInverseC(chains_nsmall)
    
    fig,axs = plt.subplots(1,3,figsize = (16,7.2))
    mkwdth = 1.5
    mkw_c1 = mkwdth
    markersize_c1 = 8
    dic_cl = {'C0':newcolorspal[3], 'C2':newcolorspal[0],'C3':newcolorspal[2],'C4':newcolorspal[4]}
    
    axs[2].plot(eigenvalues_icod_th,'d',color =newcolorspal[5], markeredgewidth = mkw_c1, markersize = markersize_c1)
    for idx, ev in enumerate(egAB_sorted_th):
        axs[2].plot(idx,ev,'o',color =dic_cl[col_ord_th[idx]], markeredgewidth = mkwdth)
    
    
    axs[2].set_xlabel('Rank')
    axs[0].set_ylabel('Eigenvalues')
    
    axs[0].plot(eigenvalues_icod_dataNsmall,'d',color =dic_cl['C0'], markeredgewidth = mkw_c1, markersize = markersize_c1)
    for idx, ev in enumerate(egAB_sorted_dataNsmall):
        axs[0].plot(idx,ev,'o',color =dic_cl[col_ord_dataNsmall[idx]], markeredgewidth = mkwdth)

    axs[0].set_xlabel('Rank')

    axs[1].plot(eigenvalues_icod_dataNbig,'d',color =dic_cl['C0'], markeredgewidth = mkw_c1, markersize = markersize_c1)
    for idx, ev in enumerate(egAB_sorted_dataNbig):
        axs[1].plot(idx,ev,'o',color =dic_cl[col_ord_dataNbig[idx]], markeredgewidth = mkwdth)
    # newcolorspal[1]
    axs[1].plot(eigenvalues_ic_dataNbig,'o',color =newcolorspal[6], markeredgewidth = mkw_c1, markersize = markersize_c1-2)
    axs[0].plot(eigenvalues_ic_dataNsmall,'o',color =newcolorspal[6], markeredgewidth = mkw_c1, markersize = markersize_c1-2,zorder = -3)
        
    axs[1].set_xlabel('Rank')
    
    axs[1].set_xlim([-5,202])
    axs[2].set_xlim([-5,202])
    axs[0].set_xlim([-5,202])
    
    axs[1].set_ylim([-1.45,13.5])
    axs[2].set_ylim([-1.45,13.5])
    axs[0].set_ylim([-1.45,13.5])
    
    l1= Line2D([0],[0],linewidth = 0,color = dic_cl['C0'],marker ='d', markeredgewidth = mkw_c1, markersize = markersize_c1)
    l2= Line2D([0],[0],linewidth = 0,color = dic_cl['C2'],marker ='o', markeredgewidth = mkwdth)
    l3= Line2D([0],[0],linewidth = 0,color = dic_cl['C3'],marker ='o', markeredgewidth = mkwdth)
    l4= Line2D([0],[0],linewidth = 0,color = newcolorspal[6],marker ='o', markeredgewidth = mkwdth,markersize = markersize_c1-2 )
    l5= Line2D([0],[0],linewidth = 0,color = newcolorspal[5],marker ='d', markeredgewidth = mkw_c1, markersize = markersize_c1)
  
    handles = ['Left and middle panels:',l4,l1,'','Right panel:',l5,'','Block diagonal approx.:',l2,l3]

    labels_data = ['',r'$(C^{-1})_{ij}$ from data       ',r'$(\tilde{C}^{-1})_{ij}$ from data','','',r'$(1-\delta_{ij})\kappa D_i D_j$' ,'','','Sector sites','Non sector sites  ']
    axs[2].legend(handles,labels_data,handletextpad=0,loc = 'upper left',bbox_to_anchor = (1.01,1))
    # axs[1].legend(handles,labels_data,handletextpad=0)
   
    axs[2].set_title('Analytical \n approximation',weight = 'bold')
    axs[0].set_title('M = 2000',weight = 'bold')
    axs[1].set_title('M = 14000',weight = 'bold')
    

    fig.subplots_adjust(wspace=0.215,hspace = 0,top= 0.880, bottom = 0.12, left = 0.06,right = 0.73)
    
    
    vmin_ic = np.min(icod)
    vmax_ic = np.max(icod)
    vmin_ic = 0
    fig,ax = plt.subplots(1,2,figsize = (16,7.2))
    ax[0].set_title(r'$\tilde{C}^{-1}_{ij}$')
    ax[0].imshow(icod,cmap = 'RdBu',vmin = vmin_ic,vmax = vmax_ic) 
    ax[1].set_title(r'Block diagonal approximation of $\tilde{C}^{-1}_{ij}$')    
    im = ax[1].imshow(approxblock,cmap = 'RdBu',vmin = vmin_ic,vmax =vmax_ic)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.025, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.subplots_adjust(wspace=0.1,hspace = 0.2,top= 0.88, bottom = 0.11, left = 0.025,right = 0.885)
  

def AnalyticalFormulaComparison_FINAL():
    
    offdiagslopes_10k_ord4 = np.load('./results/analytical_comparison/slopes_pearson/A0_05_kappaprime_logspacing_m4_m1_num20_base10_endpointTrue/slopes_order_nreal_1_4.npy')
    offdiagslopes_30k_ord4 = np.load('./results/analytical_comparison/slopes_pearson/A0_05_kappaprime_logspacing_m4_m1_num20_base10_endpointTrue/slopes_order_nreal_3_4.npy')
    offdiagslopes_50k_ord4 = np.load('./results/analytical_comparison/slopes_pearson/A0_05_kappaprime_logspacing_m4_m1_num20_base10_endpointTrue/slopes_order_nreal_5_4.npy')


    slopes_50k_ord1 = np.load('./results/analytical_comparison/slopes_pearson/slopes_order_nreal_5_1.npy')
    slopes_50k_ord2 = np.load('./results/analytical_comparison/slopes_pearson/slopes_order_nreal_5_2.npy')
    slopes_50k_ord3 = np.load('./results/analytical_comparison/slopes_pearson/slopes_order_nreal_5_3.npy')
    slopes_50k_ord4 = np.load('./results/analytical_comparison/slopes_pearson/slopes_order_nreal_5_4.npy')
        
    offdiagpearson_10k_ord4 = np.load('./results/analytical_comparison/slopes_pearson/A0_05_kappaprime_logspacing_m4_m1_num20_base10_endpointTrue/pearson_nreal_1_order_4.npy')
    offdiagpearson_30k_ord4 = np.load('./results/analytical_comparison/slopes_pearson/A0_05_kappaprime_logspacing_m4_m1_num20_base10_endpointTrue/pearson_nreal_3_order_4.npy')
    offdiagpearson_50k_ord4 = np.load('./results/analytical_comparison/slopes_pearson/A0_05_kappaprime_logspacing_m4_m1_num20_base10_endpointTrue/pearson_nreal_5_order_4.npy')

    pearson_50k_ord1 = np.load('./results/analytical_comparison/slopes_pearson/pearson_nreal_5_order_1.npy')
    pearson_50k_ord2 = np.load('./results/analytical_comparison/slopes_pearson/pearson_nreal_5_order_2.npy')
    pearson_50k_ord3 = np.load('./results/analytical_comparison/slopes_pearson/pearson_nreal_5_order_3.npy')
    pearson_50k_ord4 = np.load('./results/analytical_comparison/slopes_pearson/pearson_nreal_5_order_4.npy')
    

    kappas = np.load('./results/analytical_comparison/slopes_pearson/kappas.npy')
    kappasoffdiag = np.load('./results/analytical_comparison/slopes_pearson/A0_05_kappaprime_logspacing_m4_m1_num20_base10_endpointTrue/kappas.npy')
    
    fig,axs = plt.subplots(2,2,figsize = (16,7.2) )
    
    n_lines = 20
    mksize = 8
    # cmap = mpl.colormaps['plasma']
    cmap = mpl.cm.get_cmap('viridis')
    cmap2 = mpl.cm.get_cmap('magma')
    # Take colors at regular intervals spanning the colormap.
    colors_plasma = cmap(np.linspace(0, 1, n_lines))
    colors_viridis = cmap2(np.linspace(0, 1, n_lines))
    axs[0,0].set_title('Off diagonal $(C^{-1})_{ij}$',weight= 'bold')
    axs[0,0].plot(kappasoffdiag,offdiagslopes_10k_ord4[1,:],'-^',color = colors_plasma[16],label = 'M = 10000',markersize = mksize)
    axs[0,0].plot(kappasoffdiag,offdiagslopes_30k_ord4[1,:],'-v',color = colors_plasma[10] ,label = 'M = 30000',markersize = mksize)
    axs[0,0].plot(kappasoffdiag,offdiagslopes_50k_ord4[1,:],'-o',color = colors_plasma[3],label = 'M = 50000')
    axs[0,0].plot(kappasoffdiag,np.ones(kappas.shape),'--',color = 'k',zorder = -3)
    
    axs[0,0].set_ylabel('Slope of linear fit')
    # axs[0].set_xlabel('Selection strength $\kappa$')
    # axs[0].set_xlim([-0.2,20.2])
    axs[0,0].set_xscale('log')
    
    axs[1,0].plot(kappasoffdiag,offdiagpearson_10k_ord4[1,:],'-^',color = colors_plasma[16],label = 'M = 10000',markersize = mksize)
    axs[1,0].plot(kappasoffdiag,offdiagpearson_30k_ord4[1,:],'-v',color = colors_plasma[10],label = 'M = 30000',markersize = mksize)
    axs[1,0].plot(kappasoffdiag,offdiagpearson_50k_ord4[1,:],'-o',color = colors_plasma[3],label = 'M = 50000')
    axs[1,0].plot(kappasoffdiag,np.ones(kappas.shape),'--',color = 'k',zorder = -3)
    # axs[1].set_xlim([-0.2,20.2])
    axs[1,0].set_xscale('log')
    
    axs[1,0].set_ylabel('Pearson correlation')
    axs[1,0].set_xlabel(r'Rescaled selection strength $\tilde{\kappa}$')
    # axs[0,0].legend( bbox_to_anchor = (-0.1,1),loc='lower left',ncol = 2)
    # axs[0,0].legend()
    
    cutidx = kappas < 1
    
    idx_o1 = 17
    idx_o2 = 13
    idx_o3 = 9
    idx_o4 = 3

    axs[0,1].set_title('Diagonal $(C^{-1})_{ii}$',weight= 'bold')
    axs[0,1].plot(kappas[cutidx],slopes_50k_ord1[0,cutidx],'-s',color = colors_viridis[idx_o1],markersize = mksize,label = '$O(1)$')
    axs[0,1].plot(kappas[cutidx],slopes_50k_ord2[0,cutidx],'-^',color = colors_viridis[idx_o2],markersize = mksize,label = '$O(\kappa^2)$')
    axs[0,1].plot(kappas[cutidx],slopes_50k_ord3[0,cutidx],'-v',color = colors_viridis[idx_o3] ,markersize = mksize,label = '$O(\kappa^3)$')
    axs[0,1].plot(kappas[cutidx],slopes_50k_ord4[0,cutidx],'-o',color = colors_viridis[idx_o4],label = '$O(\kappa^4)$')
    axs[0,1].plot(kappas[cutidx],np.ones(kappas[cutidx].shape),'--',color = 'k',zorder = -3)
    
    axs[0,1].set_ylabel('Slope of linear fit')
    # axs[0].set_xlabel('Selection strength $\kappa$')
    # axs[0].set_xlim([-0.2,20.2])
    axs[0,1].set_yticks([0,1,2,3])
    axs[0,1].set_xscale('log')
    

    axs[1,1].plot(kappas[cutidx],pearson_50k_ord1[0,cutidx],'-s',color = colors_viridis[idx_o1],markersize = mksize,label = 'O(1)')    
    axs[1,1].plot(kappas[cutidx],pearson_50k_ord2[0,cutidx],'-^',color = colors_viridis[idx_o2],markersize = mksize,label = 'O($\kappa^2$)')
    axs[1,1].plot(kappas[cutidx],pearson_50k_ord3[0,cutidx],'-v',color = colors_viridis[idx_o3],markersize = mksize,label = 'O($\kappa^3$)')
    axs[1,1].plot(kappas[cutidx],pearson_50k_ord4[0,cutidx],'-o',color = colors_viridis[idx_o4],label = 'O($\kappa^4$)')
    axs[1,1].plot(kappas[cutidx],np.ones(kappas[cutidx].shape),'--',color = 'k',zorder = -3)
    # axs[1].set_xlim([-0.2,20.2])
    axs[1,1].set_xscale('log')
    
    axs[1,1].set_ylabel('Pearson correlation')
    axs[1,1].set_xlabel(r'Rescaled selection strength $\tilde{\kappa}$')
    
    labels = ['','','M = 10000','M = 30000','M = 50000','','','','','$n =$ 0','$n =$ 2','$n =$ 3','$n =$ 4']
    
    lw = 2
    l1= Line2D([0],[0],linewidth = lw,color =  colors_plasma[16],marker ='^',markersize = mksize)
    l2= Line2D([0],[0],linewidth = lw,color =  colors_plasma[10],marker ='v',markersize = mksize)
    l3= Line2D([0],[0],linewidth = lw,color =  colors_plasma[3],marker ='o')
    
    l4= Line2D([0],[0],linewidth = lw,color =  colors_viridis[idx_o1],marker ='s',markersize = mksize)
    l5= Line2D([0],[0],linewidth = lw,color =  colors_viridis[idx_o2],marker ='^',markersize = mksize)
    l6= Line2D([0],[0],linewidth = lw,color =  colors_viridis[idx_o3],marker ='v',markersize = mksize)
    l7= Line2D([0],[0],linewidth = lw,color =  colors_viridis[idx_o4],marker ='o')

    handles = ['\n'+'Left panels,\nup to order'+r' $\tilde{\kappa}^4$:','',l1,l2,l3,'','','\nRight panels,\nM = 50000,\nup to order'+r' $\tilde{\kappa}^n$:','',l4,l5,l6,l7]
    
    axs[0,1].legend(handles,labels, bbox_to_anchor = (1.05,1.1),loc='upper left',borderpad = 1)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.29,hspace = 0.32,top= 0.897, bottom = 0.142, left = 0.089,right = 0.775)



    
def RunRecoveryAUC_Allmethods_exampledata():
    
    patheq_folder = './example_generated_data/EQ_K10_alpham300_300_10real/'
    path_m5_folder = './example_generated_data/phylogeny_Alpham300_300_K10_M5/'

    parametername='Alpha'
    valname1 = 'Kappa'
    valname2 = 'Mutations'
    pseudocount = 0.00001
    
    ###############################################################################    
    ##  COMPUTE COV, ICOD RECOVERY FOR DATA WITH PHYLOGENY AND EQ
    ###############################################################################    
         
    C_lmax,C_lmin, icod_Lmax,icod_Lmin,parameters,val1,val2 = Averaging2(path_m5_folder, parametername, valname1,valname2, pseudocount)
    
    savepth ='./results/recovery/phylogeny_Alpham300_300_K10_M5/'
    
    if not(os.path.exists(savepth)):
        os.mkdir(savepth)
    
    np.save(savepth+'C_lmax.npy',C_lmax)
    np.save(savepth+'C_lmin.npy',C_lmin)
    np.save(savepth+'ICOD_Lmax.npy',icod_Lmax)
    np.save(savepth+'ICOD_Lmin.npy',icod_Lmin)
    

    C_lmax,C_lmin, icod_Lmax,icod_Lmin,parameters,val = Averaging(patheq_folder,'Alpha', 'Kappa', pseudocount)
    savepth ='./results/recovery/EQ_K10_alpham300_300_10real/'
    if not(os.path.exists(savepth)):
        os.mkdir(savepth)
    
    np.save(savepth+'C_lmax.npy',C_lmax)
    np.save(savepth+'C_lmin.npy',C_lmin)
    np.save(savepth+'ICOD_Lmax.npy',icod_Lmax)
    np.save(savepth+'ICOD_Lmin.npy',icod_Lmin)
    
    ###############################################################################    
    ##  COMPUTE COV, ICOD AUC FOR DATA WITH PHYLOGENY AND EQ
    ###############################################################################    
      
    C_lmax,C_lmin, icod_Lmax,icod_Lmin,parameters,val1,val2 = Averaging2AUC(path_m5_folder, parametername, valname1,valname2, pseudocount)
    
    savepth ='./results/AUC_sym/phylogeny_Alpham300_300_K10_M5/'
    
    if not(os.path.exists(savepth)):
        os.mkdir(savepth)
    
    np.save(savepth+'C_lmax.npy',C_lmax)
    np.save(savepth+'C_lmin.npy',C_lmin)
    np.save(savepth+'ICOD_Lmax.npy',icod_Lmax)
    np.save(savepth+'ICOD_Lmin.npy',icod_Lmin)
    

    C_lmax,C_lmin, icod_Lmax,icod_Lmin,parameters,val = AveragingAUC(patheq_folder,'Alpha', 'Kappa', pseudocount)
    savepth ='./results/AUC_sym/EQ_K10_alpham300_300_10real/'
    
    if not(os.path.exists(savepth)):
        os.mkdir(savepth)
    np.save(savepth+'C_lmax.npy',C_lmax)
    np.save(savepth+'C_lmin.npy',C_lmin)
    np.save(savepth+'ICOD_Lmax.npy',icod_Lmax)
    np.save(savepth+'ICOD_Lmin.npy',icod_Lmin)

    ###############################################################################    
    ##  COMPUTE SCA RECOVERY FOR DATA WITH PHYLOGENY AND EQ
    ###############################################################################    

    SCA_lmax,SCA_lmin,parameters,val1,val2 = Averaging2SCA(path_m5_folder, parametername, valname1,valname2, pseudocount)  
    savepth ='./results/recovery/phylogeny_Alpham300_300_K10_M5/'
    
    np.save(savepth+'SCA_lmax.npy',SCA_lmax)
    np.save(savepth+'SCA_lmin.npy',SCA_lmin)

    SCA_lmax,SCA_lmin,parameters,val = AveragingSCA(patheq_folder,'Alpha', 'Kappa', pseudocount)
    savepth ='./results/recovery/EQ_K10_alpham300_300_10real/'
    
    np.save(savepth+'SCA_lmax.npy',SCA_lmax)
    np.save(savepth+'SCA_lmin.npy',SCA_lmin)

    ###############################################################################    
    ##  COMPUTE SCA AUC FOR DATA WITH PHYLOGENY AND EQ
    ###############################################################################    

    SCA_lmax,SCA_lmin,parameters,val1,val2 = Averaging2AUCSCA(path_m5_folder, parametername, valname1,valname2, pseudocount)
    
    savepth ='./results/AUC_sym/phylogeny_Alpham300_300_K10_M5/'
    
    np.save(savepth+'SCA_lmax.npy',SCA_lmax)
    np.save(savepth+'SCA_lmin.npy',SCA_lmin)
    

    SCA_lmax,SCA_lmin,parameters,val = AveragingAUCSCA(patheq_folder,'Alpha', 'Kappa', pseudocount)
    savepth ='./results/AUC_sym/EQ_K10_alpham300_300_10real/'
    
    np.save(savepth+'SCA_lmax.npy',SCA_lmax)
    np.save(savepth+'SCA_lmin.npy',SCA_lmin)
    
    ###############################################################################    
    ##  COMPUTE CONSERVATION REC / AUC FOR DATA WITH PHYLOGENY AND EQ
    ###############################################################################    

    rec_cons,auc_cons,parameters,val1,val2 = Averaging2Conservation(path_m5_folder,parametername,valname1,valname2)
    savepth_auc ='./results/AUC_sym/phylogeny_Alpham300_300_K10_M5/'
    savepth_rec ='./results/recovery/phylogeny_Alpham300_300_K10_M5/'
    
    np.save(savepth_rec+'conservation.npy',rec_cons)
    np.save(savepth_auc+'conservation.npy',auc_cons)
    
    rec_cons,auc_cons,parameters,val = AveragingConservation(patheq_folder,'Alpha', 'Kappa')
   
    savepth_auc ='./results/AUC_sym/EQ_K10_alpham300_300_10real/'
    savepth_rec ='./results/recovery/EQ_K10_alpham300_300_10real/'
    
    np.save(savepth_rec+'conservation.npy',rec_cons)
    np.save(savepth_auc+'conservation.npy',auc_cons)



if __name__ == '__main__':
    
    
    # RunRecoveryAUC_Allmethods_exampledata() #-> example function to run ICOD,COV,SCA and conservation on data.
    
    ##list of figures
    BlockMatrixFinal_MAINversion()
    PlotEigenvaluesSuperpositionPurePhylogenySelection_INSET_subplotsSCA()
    PlotRecvsM()
    PlotRecvsM_SCA()
    PlotRECVSAlphaEQM2panels()
    PlotAUCVSAlphaEQM2panels()
    PlotRECVSAlphaEQM2panels_oppositeSPECTRUM()
    PlotAUCVSAlphaEQM2panels_oppositeSPECTRUM()
    PlotRecVSKappaEQMICOD_SUBPLOTSFINAL()
    PlotRecAUCVSKappaEQMICOD_SUBPLOTSFINAL_SCA()
    # CorrelationGscoresEigenvectors_final()
    # CorrelationGscoresCovariance_final()
    PlotREC_FULLSPECTRUM_subplots()
    PlotExtremeTSpectrumEV()
    AnalyticalFormulaComparison_FINAL()
