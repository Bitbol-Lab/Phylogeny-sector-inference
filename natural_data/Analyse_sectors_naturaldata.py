#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse natural data
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
# import utils as u
import os
from tqdm import tqdm
import scipy.io
import random
import pandas as pd
from collections import Counter
import matplotlib.colors as mcolors
import scipy.stats
import sklearn
import re
from scipy.stats import norm
import math
from scipy.optimize import curve_fit
import pdb
import pandas
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
fontsize_test = 20
colorspal6 = [(64, 83, 211), (221, 179, 16), (181, 29, 20), (0, 190, 255), (251, 73, 176), (0, 178, 93), (202, 202, 202)]
colorspal12 = [(235, 172, 35), (184, 0, 88), (0, 140, 249), (0, 110, 0), (0, 187, 173), (209, 99, 230), (178, 69, 2), (255, 146, 135), (89, 84, 214), (0, 198, 248), (135, 133, 0), (0, 167, 108), (189, 189, 189)]
newcolorspal =[]
for t in colorspal6:
    newcolorspal.append(tuple(ti/255 for ti in t))

newcolorspal12 =[]
for t in colorspal12:
    newcolorspal12.append(tuple(ti/255 for ti in t))
plt.rcParams.update({'figure.max_open_warning': 0})
# fontsize_test = 25
plt.rcParams.update({'font.size': fontsize_test})
# fig, axs = plt.subplots(3, 3)
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


def OpenSectors(path,nbr):
    return np.flip(np.load(path)[:,-nbr:],axis = 1)

def Matchnames_xcl(path,keysheet):
    df = pd.read_excel(path, sheet_name=keysheet)
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
        if isinstance(key, str):
            protfamily.append(protnames[idx])
            datasetnames.append(key)
            dictionary[key] = {'Protein':protnames[idx]}
    return dictionary

def EV_vs_Cutoff(path,idx_select, path2ndpart):
    list_files = os.listdir(path)
    for f in list_files:
        if f.startswith('.'):
            list_files.remove(f)
 
    number_sect = len(list_files)
    sectors = OpenSectors(path+list_files[0]+path2ndpart,1)
    nbrpos = sectors.shape[0]
    sector_toplot = np.empty((nbrpos,number_sect))
    list_files.sort()
    
    for idxf,f in enumerate(list_files):
     
        sector_toplot[:,idxf] = OpenSectors(path+f+path2ndpart,idx_select+1)[:,idx_select]
    
    return sector_toplot

def Conservation_vs_Cutoff(path):
    list_files = os.listdir(path)
    for f in list_files:
        if f.startswith('.'):
            list_files.remove(f)
            
            
    number_sect = len(list_files)
    sectors = np.load(path+'/'+list_files[0]+'/Conservation/conservation.npy')
    nbrpos = sectors.shape[0]
    sector_toplot = np.empty((nbrpos,number_sect))
    list_files.sort()
    
    for idxf,f in enumerate(list_files):
     
        sector_toplot[:,idxf] = np.load(path+'/'+f+'/Conservation/conservation.npy')
    
    return sector_toplot

def ComputeAUCTP(matrix,minenrichment,thr_GT):
     
      nbrres_oi,groundtruth = GroundTruth(minenrichment, thr_GT)
      lengthvec,nbrvectors = matrix.shape
      auc_vect = np.empty(nbrvectors)
      tpfrac_vect = np.empty(nbrvectors)
      
      vecbase = matrix[:,0].copy()

      fam1 = []
      fam2 = []
      fam1.append(vecbase)
      fam2.append(vecbase*-1)
      
      for i in range(1,nbrvectors):
          pcorr,_ =  scipy.stats.pearsonr(vecbase,matrix[:,i])
          if pcorr > 0:
              fam1.append(matrix[:,i])

          else:
              fam1.append(matrix[:,i]*-1)

      for idx,v in enumerate(fam1):

            tpfrac_vect[idx],_,_,auc_vect[idx] =  ComputeFraction(v,nbrres_oi,groundtruth)
            
      return tpfrac_vect,auc_vect   
  
def ComputeAUCTP_Sum(matrix,minenrichment,thr_GT):
     
      nbrres_oi,groundtruth = GroundTruth(minenrichment, thr_GT)
      lengthvec,nbrvectors = matrix.shape
      vecbase = matrix[:,0].copy()
      fam1 = []

      fam1.append(vecbase)

      
      for i in range(1,nbrvectors):
          pcorr,_ =  scipy.stats.pearsonr(vecbase,matrix[:,i])
          if pcorr > 0:
              fam1.append(matrix[:,i])

          else:
              fam1.append(matrix[:,i]*-1)

      assert np.array(fam1).shape[0] == nbrvectors
      
      ev_sum = np.sum(np.array(fam1),axis = 0)


      tpfrac_vect,_,_,auc_vect =  ComputeFraction(ev_sum,nbrres_oi,groundtruth)

      return tpfrac_vect,auc_vect   
   
def GroundTruth(min_enrichment,thr):
    blarr = min_enrichment <= thr
    idxs = np.where(min_enrichment <= thr)[0]
    number_residues = idxs.shape[0]
    return number_residues,blarr

def RemoveFilesDot(listdot):
    for f in listdot:
        if f.startswith('.'):
            listdot.remove(f)
    return listdot

def ComputeScoresSectors(pathfam,path2ndpart,methodstr,nbrsectors,bl_maxorsum):
    auc_sectors = []
    tp_sectors = []
    pearsonl = []
    spearmanl = []


    for i in range(0,nbrsectors):
        sector = EV_vs_Cutoff(pathfam+'/FinalMSAs/cutoffs/',i,path2ndpart)
        if bl_maxorsum:
            dms,cutoff_dms,idxs_del = NewDMS_cutoff(pathfam)
            if idxs_del:
                sector = np.delete(sector,idxs_del,axis = 0)

            tpfrac_vect,auc_vect = ComputeAUCTP(sector,dms,cutoff_dms)

        
        else:
            dms,cutoff_dms,idxs_del = NewDMS_cutoff(pathfam)
            if idxs_del:
                sector= np.delete(sector,idxs_del,axis = 0)

            tpfrac_vect,auc_vect = ComputeAUCTP_Sum(sector,dms,cutoff_dms)

        
        auc_sectors.append(auc_vect)
        tp_sectors.append(tpfrac_vect)


    return auc_sectors, tp_sectors

def ComputeScoresConservation(pathfam,bl_cutoffstd):
    pathconservation = pathfam+'/FinalMSAs/cutoffs'
    sector =  Conservation_vs_Cutoff(pathconservation)

    if bl_cutoffstd:
        dms,cutoff_dms,idxs_del = NewDMS_cutoff(pathfam)
        if idxs_del:
            sector = np.delete(sector,idxs_del,axis = 0)
        tpfrac_vect,auc_vect = ComputeAUCTP(sector,dms,cutoff_dms)

    else:
        dms,cutoff_dms,idxs_del = NewDMS_cutoff(pathfam)
        if idxs_del:
            sector= np.delete(sector,idxs_del,axis = 0)
        tpfrac_vect,auc_vect = ComputeAUCTP_Sum(sector,dms,cutoff_dms)

    return auc_vect,tpfrac_vect
   
def ScoresMethods(pth_family,methodstr,nbrsectors,bl_maxorsum,blapc):

      if blapc:
          path_apc = '/'+methodstr + '/APC/eigenvectors.npy'
      else:
          path_apc = '/'+methodstr + '/NO_APC/eigenvectors.npy'

      auc, tp = ComputeScoresSectors(pth_family,path_apc,methodstr,nbrsectors,bl_maxorsum)

      return auc
 
def gauss(x,mu0,sigma0):
    return 1/(sigma0*np.sqrt(2*np.pi))*np.exp(-(x-mu0)**2/2/sigma0**2)

def bimodal(x,A1,mu1,sigma1,A2,mu2,sigma2):
    return A1*gauss(x,mu1,sigma1)+A2*gauss(x,mu2,sigma2)
       
def NewDMS_cutoff(pathfam):
    
    family = pathfam.split('/')[-1]
    dms_aligned = np.load(pathfam+'/dms_aligned.npy')* dictionary_dms[family]['sign']
    
    
    if dictionary_dms[family]['bl_specific'] == True and family == 'MTH3_HAEAESTABILIZED_Tawfik2015':
        dms_aligned[dms_aligned == 0] = np.nanmax(dms_aligned)
    
    if dictionary_dms[family]['bl_specific'] == True and family == 'KKA2_KLEPN_Mikkelsen2014':
        dms_aligned = np.log10(dms_aligned)
        
    if dictionary_dms[family]['bl_specific'] == True and family == 'PA_FLU_Sun2015':
        dms_aligned = np.log10(dms_aligned)
            
    if dictionary_dms[family]['bl_specific'] == True and family == 'PABP_YEAST_Fields2013-singles':
        dms_aligned = np.log2(dms_aligned)
        
    if -np.inf in dms_aligned:
        dms_aligned[dms_aligned == -np.inf] = np.nan
        
    dms_min = np.nanmin(dms_aligned,axis = 0)
    if np.where(np.isnan(dms_min)==True)[0].size:
        idxs_del_nans = np.where(np.isnan(dms_min)==True)
        dms_min = np.delete(dms_min,idxs_del_nans)
    else:
        idxs_del_nans= []
       
    hist, bin_edges = np.histogram(dms_min,bins=20,density = True)

    n = len(hist)
    x_hist=np.zeros((n),dtype=float) 
    for ii in range(n):
        x_hist[ii]=(bin_edges[ii+1]+bin_edges[ii])/2
        
    y_hist=hist
    ymin = 0
    ymax = np.max(y_hist)
    param_optimised,_ = curve_fit(gauss,x_hist,y_hist,p0=[np.mean(dms_min),np.std(dms_min)],maxfev=10000)

    cutoff_dms = param_optimised[0]

    if dictionary_dms[family]['shape_bimodal']['bl']:
        
        if dictionary_dms[family]['shape_bimodal']['boundx_bl']:
            xbound = dictionary_dms[family]['shape_bimodal']['xbound']
            paraminit = dictionary_dms[family]['shape_bimodal']['param']
            idx_kp = x_hist < xbound
            x_hist = x_hist[idx_kp]
            y_hist = y_hist[idx_kp]
            param_optimised_bimodal,_ = curve_fit(bimodal,x_hist,y_hist,p0=paraminit,maxfev=100000)

        else:
            paraminit = dictionary_dms[family]['shape_bimodal']['param']
            param_optimised_bimodal,_ = curve_fit(bimodal,x_hist,y_hist,p0=paraminit,maxfev=100000)

    x = np.linspace(np.min(dms_min),np.max(dms_min), 1000)

    if dictionary_dms[family]['shape_bimodal']['bl']:
        y_bim = bimodal(x,*param_optimised_bimodal)

        mu1 = param_optimised_bimodal[1];mu2 = param_optimised_bimodal[4]
        idx1 = np.argmin(abs(x-mu1));idx2 =np.argmin(abs(x-mu2))
        
        idx_min = idx1 + np.argmin(bimodal(x,*param_optimised_bimodal)[idx1:idx2])

        cutoff_dms = x[idx_min]
    
    return dms_min,cutoff_dms,idxs_del_nans

def ScoresAll(dictionary):
    path_fam = './MSAs/'
    list_fam = RemoveFilesDot(os.listdir(path_fam))

    prot_names = [];
    
    icod_max = np.zeros(len(list_fam))
    mi_max = np.zeros(len(list_fam))
    sca_max = np.zeros(len(list_fam))

    icod_sum = np.zeros(len(list_fam))
    mi_sum = np.zeros(len(list_fam))
    sca_sum = np.zeros(len(list_fam))

    conservation_score_max = np.zeros(len(list_fam))
    conservation_score_sum = np.zeros(len(list_fam))
    
    pnames = []
    bimodal_bl = []
    
    for idx,family in tqdm(enumerate(list_fam)):

        pth_family = path_fam+family
        
        pnames.append(family)

        auc_icod = ScoresMethods(pth_family,'ICOD',10,True,True)
        auc_icod_sum = ScoresMethods(pth_family,'ICOD',10,False,True)

        bimodal_bl.append(dictionary_dms[family]['shape_bimodal']['bl'])
        
        auc_mi = ScoresMethods(pth_family,'MI',10,True,True)
        auc_mi_sum = ScoresMethods(pth_family,'MI',10,False,True)

        auc_sca = ScoresMethods(pth_family,'SCA',10,True,False)
        auc_sca_sum = ScoresMethods(pth_family,'SCA',10,False,False)

        aucsym = 2*np.abs(np.array(auc_icod)[0,:]-0.5)
        icod_max[idx] = np.max(aucsym)

        auc_mi_sym = 2*np.abs(np.array(auc_mi)[0,:]-0.5)
        mi_max[idx] = np.max(np.array(auc_mi_sym))
        
        auc_sca_sym = 2*np.abs(np.array(auc_sca)[0,:]-0.5)
        sca_max[idx] = np.max(np.array(auc_sca_sym))
        
        icod_sum[idx] = np.array(auc_icod_sum)[0]
        mi_sum[idx] = np.array(auc_mi_sum)[0]
        sca_sum[idx] = np.array(auc_sca_sum)[0]

        auc_conservation_sum,_ = ComputeScoresConservation(pth_family,False)
        auc_conservation,_ = ComputeScoresConservation(pth_family,True)

        auc_cons_sym = 2*np.abs(np.array(auc_conservation)-0.5)
        conservation_score_max[idx] = np.max(auc_cons_sym)
        conservation_score_sum[idx] = auc_conservation_sum
    
    
    # savepth = './results/AUC/' 
    # np.save(savepth+'icod_sum.npy',icod_sum)
    # np.save(savepth+'mi_sum.npy',mi_sum)
    # np.save(savepth+'sca_sum.npy',sca_sum)
    # np.save(savepth+'conservation_sum.npy',conservation_score_sum)
    
    # np.save(savepth+'icod_max.npy',icod_max)
    # np.save(savepth+'mi_max.npy',mi_max)
    # np.save(savepth+'sca_max.npy',sca_max)
    # np.save(savepth+'conservation_max.npy',conservation_score_max)
    
    # np.save(savepth+'protein_names.npy',pnames)
    # np.save(savepth+'bimodal_bool_shapes.npy',bimodal_bl)
    
    
def ComputeFraction(sector,number_res,blarr):
    idxs_sect= np.flip(np.argsort(sector))[:number_res]
    a = np.zeros(sector.shape,dtype = bool)
    a[idxs_sect] = True
    # import pdb;pdb.set_trace()
    tpfraction_thr = np.sum(np.logical_and(a,blarr))/number_res
    
    fpr,tpr,_ = sklearn.metrics.roc_curve(blarr, sector)
    
    auc = sklearn.metrics.auc(fpr, tpr)
    
    return tpfraction_thr,fpr,tpr,auc

def SitewiseComparison(dictionary):
    path_fam = './MSAs/'
    list_fam = RemoveFilesDot(os.listdir(path_fam))
    prot_names = []
    path_sum_icod = './sectors_icod/'

    common_sites = []
    icod_sites = []
    conservation_sites = []
    no_mthds = []
    sector_size = []
    proteinlength = []
    
    for idx,family in tqdm(enumerate(list_fam)):
        pth_family = path_fam+family
        prot_names.append(family)

        sector_sum_icod = np.load(path_sum_icod+family+'/ev0_sum.npy')        
        conservation_sum = np.load('./conservation_sum/'+family+'/conservationsum.npy')
        print(family)
        dms,cutoff_dms,idxs_del = NewDMS_cutoff(pth_family)
        
        # if idxs_del:
        #     sectormax = np.delete(sectormax,idxs_del,axis = 0)    
        #     conservation = np.delete(conservation,idxs_del,axis = 0)    
            
        
        nbrresidues,groundtruth = GroundTruth(dms, cutoff_dms)

        
        _,_,_,auc_vect =  ComputeFraction(sector_sum_icod,nbrresidues,groundtruth)
                
        _,_,_,auc_vectm1 =  ComputeFraction(sector_sum_icod*-1,nbrresidues,groundtruth)
        
        
        if auc_vect > auc_vectm1:
            signmul = 1
        else:
            signmul = -1
        
        ind_sector = np.flip(np.argsort(sector_sum_icod*signmul))

        ind_conservation = np.flip(np.argsort(conservation_sum))
        
        tp_comparison_val, labels = TP_sitecomparison(ind_sector[:nbrresidues], ind_conservation[:nbrresidues], groundtruth)
        val, cts = np.unique(tp_comparison_val,return_counts = True)

        common_sites.append(cts[val == 0])
        icod_sites.append(cts[val == 1])
        conservation_sites.append(cts[val == 2])
        no_mthds.append(cts[val == 3])
        sector_size.append(nbrresidues)
        
        assert sector_sum_icod.shape == conservation_sum.shape
        assert sector_sum_icod.shape == dms.shape
       
        proteinlength.append(dms.shape)
        
 
    
    # pt = './results/Sites_ICOD_Conservation/'
    # np.save(pt + 'icodsites.npy',icod_sites)
    # np.save(pt + 'commonsites.npy',common_sites)
    # np.save(pt + 'conservationsites.npy',conservation_sites)
    # np.save(pt + 'no_mthds.npy',no_mthds)
    # np.save(pt + 'protein_length.npy',proteinlength)
    # np.save(pt + 'sectorsize.npy',sector_size)
    
    
def TP_sitecomparison(ind_sector,ind_conservation,groundtruth):
    
    ind_gt = np.where(groundtruth == True)[0]

    val = np.empty(ind_gt.shape[0])

    for index, site in enumerate(ind_gt):
        if site in ind_sector and site in ind_conservation:
            val[index] = 0
        elif site in ind_sector and not(site in ind_conservation):
             val[index] = 1
        elif not(site in ind_sector) and site in ind_conservation:
            val[index] = 2
        elif not(site in ind_sector) and not(site in ind_conservation):
            val[index] = 3
        else:
            print('error')
    return val,['both','ICOD','conservation','none']


def auc_sym(auc):
    return 2*np.abs(auc-0.5)

def PlotAUC():
    pth = './results/AUC/'

    icodsum = np.load(pth + 'icod_sum.npy')


    idxs_sum = np.flip( np.argsort(auc_sym(icodsum)))

    icodsum = icodsum[idxs_sum]
    scasum = np.load(pth + 'sca_sum.npy')[idxs_sum]
    misum = np.load(pth + 'mi_sum.npy')[idxs_sum]

    conservation_sum = np.load(pth+'conservation_sum.npy')[idxs_sum]
    
    icodmax = np.load(pth+'icod_max.npy')[idxs_sum]
    scamax = np.load(pth+'sca_max.npy')[idxs_sum]
    mimax = np.load(pth+'mi_max.npy')[idxs_sum]

    conservation_max = np.load(pth+'conservation_max.npy')[idxs_sum]

    x = np.arange(idxs_sum.shape[0])+1

    bimodal_shapes_bol = np.load(pth+'bimodal_bool_shapes.npy')
    bimodal_shapes_bol = bimodal_shapes_bol[idxs_sum]
    
    inverse_bim = [not t for t in bimodal_shapes_bol]
    fig = plt.figure(figsize =(16,7.2))
    
    hval = 0.8
    plt.bar(x[inverse_bim],hval*np.ones(x.shape)[inverse_bim],color = 'lightgrey',alpha = 0.5,label ='Unimodal')
    
    plt.plot(x,auc_sym(icodsum),'o',color = newcolorspal[0],label = 'ICOD')
    
    plt.plot(x,auc_sym(scasum),'s',color = newcolorspal[1],label = 'SCA')
 
    plt.plot(x,auc_sym(misum),'^',color = newcolorspal[3],label = 'MI')

    plt.plot(x,auc_sym(conservation_sum),'d',color =  newcolorspal12[6],label ='Conservation')
    plt.plot(x,auc_sym(0.5*np.ones(x.shape)),':',color = 'k',label ='Null model')


    plt.xlabel('Protein family')
    plt.xticks([1,5,10,15,20,25,30])
    plt.ylabel('Symmetrized AUC')
    plt.legend(bbox_to_anchor = (1.05, 1), loc='upper left')
    plt.xlim([-0.5,31])
    plt.ylim([-0.01,0.8])
    fig.subplots_adjust(wspace=0.2,hspace = 0.2,top= 0.952, bottom = 0.163, left = 0.083,right = 0.707)

    print('icod sum: {:.2f}'.format(np.mean(auc_sym(icodsum))))
    print('sca sum: {:.2f}'.format(np.mean(auc_sym(scasum))))
    print('mi sum: {:.2f}'.format(np.mean(auc_sym(misum))))
    print('Conservation sum: {:.2f}'.format(np.mean(auc_sym(conservation_sum))))

    fig = plt.figure(figsize =(16,7.2))
    
    hval = 0.85
    plt.bar(x[inverse_bim],hval*np.ones(x.shape)[inverse_bim],color = 'lightgrey',alpha = 0.5,label ='Unimodal')
    
    plt.plot(x,icodmax,'o',color = newcolorspal[0],label = 'ICOD max')
   
    plt.plot(x,scamax,'s',color = newcolorspal[1],label = 'SCA max')

    plt.plot(x,mimax,'^',color = newcolorspal[3],label = 'MI max')

    plt.plot(x,conservation_max,'d',color =  newcolorspal12[6],label ='Conservation max')
    plt.plot(x,auc_sym(0.5*np.ones(x.shape)),':',color = 'k',label ='Null model')

    plt.xlabel('Protein family')
    plt.xticks([1,5,10,15,20,25,30])
    plt.ylabel('Symmetrized AUC')
    plt.legend(bbox_to_anchor = (1.05, 1), loc='upper left')
    plt.xlim([-0.5,31])
    plt.ylim([-0.01,0.85])
    fig.subplots_adjust(wspace=0.2,hspace = 0.2,top= 0.952, bottom = 0.163, left = 0.083,right = 0.707)

    print('icod max: {:.2f}'.format(np.mean(icodmax)))
    print('sca max:{:.2f}' .format(np.mean(scamax)))
    print('mi max:{:.2f}' .format(np.mean(mimax)))
    print('Conservation max:{:.2f}'.format(np.mean(conservation_max)))
    
def RemoveFilesDot(listdot):
    for f in listdot:
        if f.startswith('.'):
            listdot.remove(f)
    return listdot

def gauss(x,mu0,sigma0):
    # print(mu0)
    return 1/(sigma0*np.sqrt(2*np.pi))*np.exp(-(x-mu0)**2/2/sigma0**2)

def bimodal(x,A1,mu1,sigma1,A2,mu2,sigma2):
    return A1*gauss(x,mu1,sigma1)+A2*gauss(x,mu2,sigma2)

def ComputeOnefamily(family):
    path_fam = './MSAs'
    list_fam = RemoveFilesDot(os.listdir(path_fam))

    path_dmsfam = path_fam+'/'+family
    dms_aligned = np.load(path_dmsfam+'/dms_aligned.npy') * dictionary_dms[family]['sign']
  
    if dictionary_dms[family]['bl_specific'] == True and family == 'MTH3_HAEAESTABILIZED_Tawfik2015':
        dms_aligned[dms_aligned == 0] = np.nanmax(dms_aligned)
    
    if dictionary_dms[family]['bl_specific'] == True and family == 'KKA2_KLEPN_Mikkelsen2014':
        dms_aligned = np.log10(dms_aligned)
        
    if dictionary_dms[family]['bl_specific'] == True and family == 'PA_FLU_Sun2015':
        dms_aligned = np.log10(dms_aligned)
    
    if -np.inf in dms_aligned:
        dms_aligned[dms_aligned == -np.inf] = np.nan
        
    dms_min = np.nanmin(dms_aligned,axis = 0)
    if np.where(np.isnan(dms_min)==True)[0].size:
        idxs_del_nans = np.where(np.isnan(dms_min)==True)
        dms_min = np.delete(dms_min,idxs_del_nans)
    else:
        idxs_del_nans= []
       
    hist, bin_edges = np.histogram(dms_min,bins=20,density = True)
  
  
    n = len(hist)
    x_hist=np.zeros((n),dtype=float) 
    for ii in range(n):
        x_hist[ii]=(bin_edges[ii+1]+bin_edges[ii])/2
        
    y_hist=hist
    ymin = 0
    ymax = np.max(y_hist)
    param_optimised,_ = curve_fit(gauss,x_hist,y_hist,p0=[np.mean(dms_min),np.std(dms_min)],maxfev=10000)
    cutoff_dms=param_optimised[0]
    if dictionary_dms[family]['shape_bimodal']['bl']:
        
        if dictionary_dms[family]['shape_bimodal']['boundx_bl']:
            xbound = dictionary_dms[family]['shape_bimodal']['xbound']
            paraminit = dictionary_dms[family]['shape_bimodal']['param']
            idx_kp = x_hist < xbound
            x_hist = x_hist[idx_kp]
            y_hist = y_hist[idx_kp]
            param_optimised,_ = curve_fit(bimodal,x_hist,y_hist,p0=paraminit,maxfev=100000)
  
        else:
            paraminit = dictionary_dms[family]['shape_bimodal']['param']
            param_optimised,_ = curve_fit(bimodal,x_hist,y_hist,p0=paraminit,maxfev=100000)
    x = np.linspace(np.min(dms_min),np.max(dms_min), 1000)
    return x,dms_min,param_optimised,dictionary_dms[family]['shape_bimodal']['bl'],ymin,ymax

def Plot_DMS(dic_protnames):
    idx_fit_gauss = 2
    idx_fit_bim = 1
    idx_ctf = 5
    lwdth = 2.5
    fig,axs = plt.subplots(2,3, figsize=(16,7.2))
    family = 'TPMT_HUMAN_Fowler2018'
    family_name = dic_protnames['TPMT_HUMAN_Fowler2018']
    
    # family_name = 'Protein family {}'.format(newdic['TPMT_HUMAN_Fowler2018']['Label'])
    x,dms_min, paramopt, bl_bim, ymin, ymax = ComputeOnefamily(family)
    axs[0,0].set_title(family_name,fontsize = fontsize_test)
    axs[0,0].hist(dms_min,color = newcolorspal[0],bins=20,density = True)
    
    if bl_bim:
        y_bim = bimodal(x,*paramopt)
        axs[0,0].plot(x,y_bim,lw=lwdth,color = newcolorspal[idx_fit_bim],linestyle= '-',label='bimodal')
        mu1 = paramopt[1];mu2 = paramopt[4]

        idx1 = np.argmin(abs(x-mu1));idx2 =np.argmin(abs(x-mu2))
        
        idx_min = idx1 + np.argmin(bimodal(x,*paramopt)[idx1:idx2])
        axs[0,0].vlines(x[idx_min],ymin,ymax,color = newcolorspal[idx_ctf],linewidth = lwdth,label = 'bimodal cutoff')
   
    else:
        axs[0,0].plot(x,gauss(x,*paramopt),color = newcolorspal[idx_fit_gauss],linestyle = '-',lw=lwdth)
        axs[0,0].vlines(paramopt[0],ymin,ymax,linewidth = lwdth,color =newcolorspal[idx_ctf],label = '$\mu$')
    
    family = 'BRCA1_HUMAN_BRCT'
    family_name = dic_protnames['BRCA1_HUMAN_BRCT']
    # family_name = 'Protein family {}'.format(newdic['BRCA1_HUMAN_BRCT']['Label'])
    x,dms_min, paramopt, bl_bim, ymin, ymax = ComputeOnefamily(family)
    axs[0,1].set_title(family_name,fontsize = fontsize_test)
    axs[0,1].hist(dms_min,color = newcolorspal[0],bins=20,density = True)
    
    if bl_bim:
        y_bim = bimodal(x,*paramopt)
        axs[0,1].plot(x,y_bim,lw=lwdth,color = newcolorspal[idx_fit_bim],linestyle= '-',label='bimodal')
        mu1 = paramopt[1];mu2 = paramopt[4]

        idx1 = np.argmin(abs(x-mu1));idx2 =np.argmin(abs(x-mu2))
        
        idx_min = idx1 + np.argmin(bimodal(x,*paramopt)[idx1:idx2])
        axs[0,1].vlines(x[idx_min],ymin,ymax,color = newcolorspal[idx_ctf],linewidth = lwdth,label = 'bimodal cutoff')
   
    else:
        axs[0,1].plot(x,gauss(x,*paramopt),linestyle = '-',color=newcolorspal[idx_fit_gauss],lw=lwdth)
        axs[0,1].vlines(paramopt[0],ymin,ymax,linewidth = lwdth,color =newcolorspal[idx_ctf],label = '$\mu$')
    family = 'DLG4_RAT_Ranganathan2012'
    family_name = dic_protnames['DLG4_RAT_Ranganathan2012']
    # family_name = 'Protein family {}'.format(newdic['DLG4_RAT_Ranganathan2012']['Label'])
    x,dms_min, paramopt, bl_bim, ymin, ymax = ComputeOnefamily(family)
    axs[0,2].set_title(family_name,fontsize = fontsize_test)
    axs[0,2].hist(dms_min,color = newcolorspal[0],bins=20,density = True)
    
    if bl_bim:
        y_bim = bimodal(x,*paramopt)
        axs[0,2].plot(x,y_bim,color = newcolorspal[idx_fit_bim],lw=lwdth,linestyle= '-',label='bimodal')
        mu1 = paramopt[1];mu2 = paramopt[4]

        idx1 = np.argmin(abs(x-mu1));idx2 =np.argmin(abs(x-mu2))
        
        idx_min = idx1 + np.argmin(bimodal(x,*paramopt)[idx1:idx2])
        axs[0,2].vlines(x[idx_min],ymin,ymax,color = newcolorspal[idx_ctf],linewidth = lwdth,label = 'bimodal cutoff')
   
    else:
        axs[0,2].plot(x,gauss(x,*paramopt),linestyle = '-',color=newcolorspal[idx_fit_gauss],lw=lwdth)
        axs[0,2].vlines(paramopt[0],ymin,ymax,linewidth = lwdth,color =newcolorspal[idx_ctf],label = '$\mu$')
    
    family = 'TPK1_HUMAN_Roth2017'
    family_name = dic_protnames['TPK1_HUMAN_Roth2017']
    # family_name = 'Protein family {}'.format(newdic['TPK1_HUMAN_Roth2017']['Label'])
    x,dms_min, paramopt, bl_bim, ymin, ymax = ComputeOnefamily(family)
    axs[1,0].set_title(family_name,fontsize = fontsize_test)
    axs[1,0].hist(dms_min,color = newcolorspal[0],bins=20,density = True)
    
    if bl_bim:
        y_bim = bimodal(x,*paramopt)
        axs[1,0].plot(x,y_bim,lw=lwdth,color = newcolorspal[idx_fit_bim],linestyle= '-',label='bimodal')
        mu1 = paramopt[1];mu2 = paramopt[4]

        idx1 = np.argmin(abs(x-mu1));idx2 =np.argmin(abs(x-mu2))
        
        idx_min = idx1 + np.argmin(bimodal(x,*paramopt)[idx1:idx2])
        axs[1,0].vlines(x[idx_min],ymin,ymax,color = newcolorspal[idx_ctf],linewidth = lwdth,label = 'bimodal cutoff')
   
    else:
        axs[1,0].plot(x,gauss(x,*paramopt),linestyle = '-',color=newcolorspal[idx_fit_gauss],lw=lwdth)
        axs[1,0].vlines(paramopt[0],ymin,ymax,linewidth = lwdth,color =newcolorspal[idx_ctf],label = '$\mu$')
    
    family = 'BLAT_ECOLX_Ranganathan2015'
    family_name = dic_protnames['BLAT_ECOLX_Ranganathan2015']
    # family_name = 'Protein family {}'.format(newdic['BLAT_ECOLX_Ranganathan2015']['Label'])
    x,dms_min, paramopt, bl_bim, ymin, ymax = ComputeOnefamily(family)
    axs[1,1].set_title(family_name,fontsize = fontsize_test)
    axs[1,1].hist(dms_min,color = newcolorspal[0],bins=20,density = True)
    
    if bl_bim:
        y_bim = bimodal(x,*paramopt)
        axs[1,1].plot(x,y_bim,lw=lwdth,color= newcolorspal[idx_fit_bim],linestyle= '-',label='bimodal')
        mu1 = paramopt[1];mu2 = paramopt[4]

        idx1 = np.argmin(abs(x-mu1));idx2 =np.argmin(abs(x-mu2))
        
        idx_min = idx1 + np.argmin(bimodal(x,*paramopt)[idx1:idx2])
        axs[1,1].vlines(x[idx_min],ymin,ymax,color = newcolorspal[idx_ctf],linewidth = lwdth,label = 'bimodal cutoff')
   
    else:
        axs[1,1].plot(x,gauss(x,*paramopt),linestyle = '-',color=newcolorspal[idx_fit_gauss],lw=lwdth)
        axs[1,1].vlines(paramopt[0],ymin,ymax,linewidth = lwdth,color =newcolorspal[idx_ctf],label = '$\mu$')
    
    family = 'B3VI55_LIPSTSTABLE'
    family_name = dic_protnames['B3VI55_LIPSTSTABLE']
    # family_name = 'Protein family {}'.format(newdic['B3VI55_LIPSTSTABLE']['Label'])
    x,dms_min, paramopt, bl_bim, ymin, ymax = ComputeOnefamily(family)
    axs[1,2].set_title(family_name,fontsize = fontsize_test)
    axs[1,2].hist(dms_min,color = newcolorspal[0],bins=20,density = True)
    
    if bl_bim:
        y_bim = bimodal(x,*paramopt)
        axs[1,2].plot(x,y_bim,color = newcolorspal[idx_fit_bim],lw=lwdth,linestyle= '-',label='bimodal')
        mu1 = paramopt[1];mu2 = paramopt[4]

        idx1 = np.argmin(abs(x-mu1));idx2 =np.argmin(abs(x-mu2))
        
        idx_min = idx1 + np.argmin(bimodal(x,*paramopt)[idx1:idx2])
        axs[1,2].vlines(x[idx_min],ymin,ymax,color = newcolorspal[idx_ctf],linewidth = lwdth,label = 'bimodal cutoff')
   
    else:
        axs[1,2].plot(x,gauss(x,*paramopt),linestyle = '-',color=newcolorspal[idx_fit_gauss],lw=lwdth)
        axs[1,2].vlines(paramopt[0],ymin,ymax,linewidth = lwdth,color =newcolorspal[idx_ctf],label = '$\mu$')
    

    extra = Rectangle((0, 0), 1, 1, fc=newcolorspal[0], linewidth=2)
    
    handles1 = [extra,
                Line2D([0],[0],linewidth = lwdth, color = newcolorspal[idx_fit_bim]),
                Line2D([0],[0],linewidth = lwdth, color = newcolorspal[idx_ctf])
        ]
    labels1 = ['Experimental value', 'Bimodal fit', 'Cutoff']
    axs[0,2].legend(handles1,labels1,loc ='upper left',bbox_to_anchor = (1.01,1))
    
    handles2 = [extra,
                Line2D([0],[0],linewidth = lwdth, color = newcolorspal[idx_fit_gauss]),
                Line2D([0],[0],linewidth = lwdth, color = newcolorspal[idx_ctf])
        ]
    labels2 = ['Experimental value', 'Gaussian fit', 'Cutoff']
    axs[1,2].legend(handles2,labels2,loc ='upper left',bbox_to_anchor = (1.01,1))
    
    axs[0,0].set_ylabel('Norm. counts')
    axs[1,0].set_ylabel('Norm. counts')

 
    
    axs[0, 0].text(-0.35, 0.25, 'Bimodal', transform=axs[0, 0].transAxes, size=fontsize_test,rotation = 'vertical',weight = 'bold')
    axs[1, 0].text(-0.35, 0.2, 'Unimodal', transform=axs[1, 0].transAxes, size=fontsize_test,rotation = 'vertical',weight = 'bold')
    
    axs[1, 1].text(0.1, -0.37, 'Min. DMS scores', transform=axs[1, 1].transAxes, size=fontsize_test)
    
    
    fig.subplots_adjust(wspace=0.32,hspace = 0.42,top= 0.895, bottom = 0.13, left = 0.07,right = 0.74)


def MappingProteinFamiliesNames():
    dic_protnames = {}
    dic_protnames['MK01_HUMAN_Johannessen'] = 'Mitogen-activated protein kinase 1'
    dic_protnames['CALM1_HUMAN_Roth2017'] = 'Calmodulin-1'
    dic_protnames['TPK1_HUMAN_Roth2017'] = 'Thiamin pyrophosphokinase 1'
    dic_protnames['SUMO1_HUMAN_Roth2017'] = 'Small ubiquitin-related modifier 1'
    dic_protnames['UBC9_HUMAN_Roth2017'] = 'SUMO-conjugating enzyme UBC9'
    dic_protnames['BRCA1_HUMAN_RING'] = 'BRCA 1 (RING domain)'
    dic_protnames['BRCA1_HUMAN_BRCT'] = 'BRCA 1 (BRCT domain)'
    dic_protnames['GAL4_YEAST_Shendure2015'] = 'GAL4 (DNA-binding domain)'
    dic_protnames['IF1_ECOLI_Kishony'] = 'Translation initiation factor IF1'
    dic_protnames['BF520_env_Bloom2018'] = 'HIV env protein (BF520)'
    dic_protnames['BG505_env_Bloom2018'] = 'HIV env protein (BG505)'
    dic_protnames['DLG4_RAT_Ranganathan2012'] = 'PSD95 (PDZ domain)'
    dic_protnames['B3VI55_LIPST_Whitehead2015'] = 'Levoglucosan kinase'
    dic_protnames['B3VI55_LIPSTSTABLE'] = 'Levoglucosan kinase (stabilized)'
    dic_protnames['RL401_YEAST_Fraser2016'] = 'Ubiquitin'
    dic_protnames['RASH_HUMAN_Kuriyan'] = 'HRas'
    dic_protnames['UBE4B_MOUSE_Klevit2013-singles'] = 'UBE4B (U-box domain)'
    dic_protnames['HG_FLU_Bloom2016'] = 'Influenza hemagglutinin'
    dic_protnames['BLAT_ECOLX_Ranganathan2015'] = 'β-lactamase'
    dic_protnames['KKA2_KLEPN_Mikkelsen2014'] = 'Kanamycin kinase APH(3\')-II'
    dic_protnames['MTH3_HAEAESTABILIZED_Tawfik2015'] = 'DNA methylase HaeIII'
    dic_protnames['TPMT_HUMAN_Fowler2018'] = 'Thiopurine S-methyltransferase'
    dic_protnames['PTEN_HUMAN_Fowler2018'] = 'PTEN'
    dic_protnames['BG_STRSQ_hmmerbit'] = 'β-glucosidase'
    dic_protnames['AMIE_PSEAE_Whitehead'] = 'Aliphatic amide hydrolase'
    dic_protnames['POLG_HCVJF_Sun2014'] = 'Hepatitis C NS5A'
    dic_protnames['YAP1_HUMAN_Fields2012-singles'] = 'YAP1 (WW domain)'
    dic_protnames['HSP82_YEAST_Bolon2016'] = 'HSP90 (ATPase domain)'
    dic_protnames['PA_FLU_Sun2015'] = 'Influenza polymerase PA subunit'
    dic_protnames['PABP_YEAST_Fields2013-singles'] = 'PABP singles (RRM domain)'
    
    pth = './results/AUC/'
    pnames = np.load(pth+'protein_names.npy')

    icodsum = np.load(pth + 'icod_sum.npy')
    idxs_sum = np.flip( np.argsort(auc_sym(icodsum)))
    pnames_order = pnames[idxs_sum]
    
    icodsum = auc_sym(icodsum[idxs_sum])
    scasum = auc_sym(np.load(pth + 'sca_sum.npy')[idxs_sum])
    misum = auc_sym(np.load(pth + 'mi_sum.npy')[idxs_sum])
    conservation_sum =auc_sym( np.load(pth+'conservation_sum.npy')[idxs_sum])
    
    icodmax = np.load(pth+'icod_max.npy')[idxs_sum]
    scamax = np.load(pth+'sca_max.npy')[idxs_sum]
    mimax = np.load(pth+'mi_max.npy')[idxs_sum]
    conservation_max = np.load(pth+'conservation_max.npy')[idxs_sum]
    
    pt = './results/Sites_ICOD_Conservation/'
    icod_sites = np.load(pt + 'icodsites.npy')[idxs_sum]
    common_sites = np.load(pt + 'commonsites.npy')[idxs_sum]
    conservation_sites = np.load(pt + 'conservationsites.npy')[idxs_sum]
    no_mthds = np.load(pt + 'no_mthds.npy')[idxs_sum]
    
    proteinlength = np.load(pt+'protein_length.npy')[idxs_sum]
    sectorsize = np.load(pt+'sectorsize.npy')[idxs_sum]
   
    newdic2 = {}
    for idx,p in enumerate(pnames_order):
        if dictionary_dms[p]['shape_bimodal']['bl']:
            dmsshape = 2
        else:
            dmsshape = 1

        newdic2[p] = {'Name':dic_protnames[p], 'Label': idx+1, 'DMS shape':dmsshape,
                     'ICOD':'{:.2f}'.format(icodsum[idx]),
                    'Conservation':'{:.2f}'.format(conservation_sum[idx]),
                    'SCA':'{:.2f}'.format(scasum[idx]),'MI':'{:.2f}'.format(misum[idx]),
                    'Common':common_sites[idx][0],'ICOD site':icod_sites[idx][0],
                    'Conservation site':conservation_sites[idx][0],
                    'Sector Size':sectorsize[idx],
                    'L':proteinlength[idx][0]}

    

    df = pandas.DataFrame.from_dict(newdic2,orient = 'index')
    print(df.to_latex(index=False,formatters={"name": str.upper},float_format="{:.2f}".format,))
    pandas.set_option('display.max_colwidth', None)
    
    
    return df.to_latex(index=False,formatters={"name": str.upper},float_format="{:.2f}".format,),dic_protnames
    

dictionary_dms ={}
dictionary_dms['MK01_HUMAN_Johannessen'] = {'sign':-1,'bl_specific':False,'shape_bimodal':{'bl':False}}
dictionary_dms['CALM1_HUMAN_Roth2017'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':False}}
dictionary_dms['TPK1_HUMAN_Roth2017'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':False}}
dictionary_dms['SUMO1_HUMAN_Roth2017'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':False}}
dictionary_dms['UBC9_HUMAN_Roth2017'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':False,'param':[3,-0.2,0.1,1.5,0,0.1]}}
dictionary_dms['BRCA1_HUMAN_RING'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':True,'boundx_bl':True,'xbound':-1,'param':[0.2, -2.4, 0.12,0.08,-1.6,0.1]}}
dictionary_dms['BRCA1_HUMAN_BRCT'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':True,'boundx_bl':True,'xbound':-1.4,'param':[0.05, -3.3, 0.2, 0.3, -2, 0.45]}}
dictionary_dms['GAL4_YEAST_Shendure2015'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':True,'boundx_bl':True,'xbound':-5,'param':[1,-14,1.2,0.12,-6,0.8]}}
dictionary_dms['IF1_ECOLI_Kishony'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':True,'boundx_bl':True,'xbound':0.5,'param':[0.35,0.03,0.1,0.35,0.3,0.1]}}
dictionary_dms['BF520_env_Bloom2018'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':True,'boundx_bl':False,'param':[0.3,-5.7,1.5,0.12,-2,0.2]}}
dictionary_dms['BG505_env_Bloom2018'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':False}}
dictionary_dms['DLG4_RAT_Ranganathan2012'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':True,'boundx_bl':False,'param':[0.9,-1.5,0.2,1,-0.2,0.2]}}
dictionary_dms['B3VI55_LIPST_Whitehead2015'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':False}}
dictionary_dms['B3VI55_LIPSTSTABLE'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':False}}
dictionary_dms['RL401_YEAST_Fraser2016'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':True,'boundx_bl':True,'xbound':-0.32, 'param':[1, -0.5, 0.01, 0.02, -0.36, 0.004]}}
dictionary_dms['RASH_HUMAN_Kuriyan'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':True,'boundx_bl':False,'param':[2,-0.7,0.2,1.2,-0.2,0.2]}}
dictionary_dms['UBE4B_MOUSE_Klevit2013-singles'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':True,'boundx_bl':True,'xbound':-0.55,'param':[0.4,-2.5,0.2,0.2,-1.3,0.2]}}
dictionary_dms['HG_FLU_Bloom2016'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':False}}
dictionary_dms['BLAT_ECOLX_Ranganathan2015'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':False}}
dictionary_dms['KKA2_KLEPN_Mikkelsen2014'] = {'sign':1,'bl_specific':True,'shape_bimodal':{'bl':True,'boundx_bl':True,'xbound':-0.51,'param':[1, -1.27, 0.2, 0.1, -0.7, 0.1]}}
dictionary_dms['MTH3_HAEAESTABILIZED_Tawfik2015'] = {'sign':1,'bl_specific':True,'shape_bimodal':{'bl':True,'boundx_bl':False,'param':[3,0.07,0.05,1.2,0.3,0.2]}}
dictionary_dms['TPMT_HUMAN_Fowler2018'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':True,'boundx_bl':False,'param':[1.2,-0.1,0.1,1.3,0.4,0.2]}}
dictionary_dms['PTEN_HUMAN_Fowler2018'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':False}}
dictionary_dms['BG_STRSQ_hmmerbit'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':True,'boundx_bl':False,'param':[0.5,-3,0.2,0.4,-0.5,0.1]}}
dictionary_dms['AMIE_PSEAE_Whitehead'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':False}}
dictionary_dms['POLG_HCVJF_Sun2014'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':False}}
dictionary_dms['YAP1_HUMAN_Fields2012-singles'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':True,'boundx_bl':True,'xbound':0.23,'param':[0.4,0.06,0.03,0.1,0.18,0.015]}}
dictionary_dms['HSP82_YEAST_Bolon2016'] = {'sign':1,'bl_specific':False,'shape_bimodal':{'bl':True,'boundx_bl':False,'param':[1,-1.2,0.2,2,-0.1,0.1]}}
dictionary_dms['PA_FLU_Sun2015'] = {'sign':1,'bl_specific':True,'shape_bimodal':{'bl':False}}
dictionary_dms['PABP_YEAST_Fields2013-singles'] = {'sign':1,'bl_specific':True,'shape_bimodal':{'bl':True,'boundx_bl':False,'param':[ 2.35018163e-01, -5.66583046e+00,  7.03163016e-01,  1.20562960e+12,1.46244393e+02,  2.06810154e+01]}}


if __name__ == '__main__':
                    
    dictionary = Matchnames_xcl('./Metadata_dms_keysheetnames.xls','data')
    
    #From the cutoff MSA compute AUC for every scores
    # ScoresAll(dictionary) #-> at the end of the function, the results can be saved in ./results/AUC
    
    #Compute the number of common sites between ICOD and conservation
    # SitewiseComparison(dictionary) #-> at the end of the function, the results can be saved in ./results/Sites_ICOD_Conservation
    
    latextable,dic_protnames = MappingProteinFamiliesNames()
    Plot_DMS(dic_protnames)
    PlotAUC()