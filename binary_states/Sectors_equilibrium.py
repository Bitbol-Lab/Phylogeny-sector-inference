#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to generate sequences with selection under trait
"""
import sys
import os
import numpy as np
import random

import time
# set seed
seed = 34
random.seed(seed)

from tqdm import tqdm
import matplotlib.pyplot as plt
from math import exp, expm1
from datetime import datetime, date
import h5py
startTime = datetime.now()
today = date.today()
from numba import jit

import numba 

def GenerateDeltaNormal(average_sector,average_nonsector, variance_sector,variance_nonsector, number_spins, number_components_sector):
    
    delta = np.zeros(number_spins)
    
    delta[0:number_components_sector] = np.random.normal(average_sector,variance_sector,number_components_sector)
    delta[number_components_sector:number_spins] = np.random.normal(average_nonsector,variance_nonsector,number_spins - number_components_sector)
    
    return delta


@jit(nopython=True, fastmath = True)#vectorize
def ComputeFitness(sequence,delta,alpha,kappa):
    
    return -kappa/2 * (-alpha + np.sum(sequence*delta))**2
    
@jit(nopython=True)
def MCEvolution(sequence, delta, alpha, kappa,number_accepted_flips, number_spins):
            
    energy = -ComputeFitness(sequence,delta,alpha,kappa)
    
    
    counter = 0
    while(counter < number_accepted_flips):

        rand_site = random.randrange(0,number_spins,1)
        deltaE = - 2*kappa * delta[rand_site]*sequence[rand_site]*((np.sum(delta*sequence)-delta[rand_site]*sequence[rand_site]) - alpha)

        if deltaE < 0:
            
            energy += deltaE
            sequence[rand_site] *= -1

            counter += 1

        
        elif random.uniform(0,1) <  np.exp(-deltaE):

            energy += deltaE
            sequence[rand_site] *= -1
            
            counter += 1

    return sequence
    
@jit(nopython=True)
def Run(matrix_chains, number_accepted_flips,number_sequences,number_spins,delta,alpha,kappa):
    
    for i in range(0,number_sequences):
        
        sequence_init = 2*(np.random.randint(0,2,number_spins))-1


        matrix_chains[i,:] = MCEvolution(sequence_init, delta, alpha, kappa,number_accepted_flips, number_spins)
    
    return matrix_chains
       



if __name__ == '__main__':
        

###############################################################################    
##  GENERATE SEQUENCES AT EQUILIBRIUM AS FUNCTION OF KAPPA
###############################################################################    
      
    
    # number_spins = 200
    # number_accepted_flips = 3000
    # number_sequences = 2048
    
    
    # average_sector = 5; variance_sector = 0.25
    # average_nonsector = 0.5; variance_nonsector = 0.25
    # number_components_sector = 20
    
    # # delta = GenerateDeltaNormal(average_sector,average_nonsector, variance_sector,variance_nonsector, number_spins, number_components_sector)
    # delta = np.load('./vector_mut_effects_2states/delta.npy')
    
    
    # kappaprimes = np.array([1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])/4
    
    # number_av = 10
    # for i in tqdm(range(0,number_av)):
        
    #     alpha = 90
        
    #     date = today.strftime("%Y_%m_%d_")
    #     hour = startTime.strftime('%H_%M_%S_')
    #     path = './example_generated_data/EQ_alpha{}_kappa0_25_25_{}real/'.format(alpha,number_av)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
            
    #     para = [number_spins, number_accepted_flips, number_sequences,number_components_sector, average_sector, variance_sector,average_nonsector,variance_nonsector, seed]
    
    #     filename = path+date+hour+'Sectors_nspins{}_flips{}_nseq{}_seed_{}_deltanormal_Kmin{}_Kmax{}_alpha{}_realnbr{}.h5'.format(number_spins, number_accepted_flips, number_sequences,seed,min(kappaprimes),max(kappaprimes),int(alpha),i)
    #     file = h5py.File(filename, 'w')
    #     file.close()
    
    #     file = h5py.File(filename, 'r+')
    #     file.create_dataset('Parameters', data = np.array(para))
    #     file.create_dataset('Delta', data = delta)
    #     file.close()
    
        
    #     final_sequences = np.zeros((kappaprimes.shape[0],number_sequences,number_spins),dtype = np.int8)
    
    #     for idxk,k in enumerate(kappaprimes):
    
    #         kappa = k/np.dot(delta,delta)
    
       
    #         matrix_chains = np.zeros([number_sequences,number_spins],dtype = np.int8)
            
    #         final_sequences[idxa,:,:] = Run(matrix_chains, number_accepted_flips,number_sequences,number_spins,delta,alpha,kappa)
            
            
    #     file = h5py.File(filename, 'r+')
    #     file.create_dataset('Chains', data = final_sequences, dtype = np.int8)
    #     file.create_dataset('Kappa', data = np.array(kappaprime))
    #     file.create_dataset('Alpha', data = np.array(alphas))
    #     file.create_dataset('NumberSequences', data = np.array(number_sequences))
    #     file.create_dataset('NumberSpins', data = np.array(number_spins))
    #     file.create_dataset('NumberFlips', data = np.array(number_accepted_flips))
    #     file.close() 


###############################################################################    
##  GENERATE SEQUENCES AT EQUILIBRIUM AS FUNCTION OF ALPHA
###############################################################################    
    
    
    number_spins = 200
    number_accepted_flips = 3000
    number_sequences = 2048
    
    
    average_sector = 5; variance_sector = 0.25
    average_nonsector = 0.5; variance_nonsector = 0.25
    number_components_sector = 20
    
    # delta = GenerateDeltaNormal(average_sector,average_nonsector, variance_sector,variance_nonsector, number_spins, number_components_sector)
    delta = np.load('./vector_mut_effects_2states/delta.npy')
    
    alphas = np.linspace(-300,300,41)
    
    number_av = 10
    for i in tqdm(range(0,number_av)):
        
        kappaprime = 10
        
        date = today.strftime("%Y_%m_%d_")
        hour = startTime.strftime('%H_%M_%S_')
        path = './example_generated_data/EQ_K{}_alpham300_300_{}real/'.format(kappaprime,number_av)
        if not os.path.exists(path):
            os.makedirs(path)
            
        para = [number_spins, number_accepted_flips, number_sequences,number_components_sector, average_sector, variance_sector,average_nonsector,variance_nonsector, seed]
        
        filename = path+date+hour+'Sectors_nspins{}_flips{}_nseq{}_seed_{}_Alphamin{}_Alphamax{}_K{}_realnbr{}.h5'.format(number_spins, number_accepted_flips, number_sequences,seed,int(np.min(alphas)),int(np.max(alphas)),int(kappaprime),i)
        file = h5py.File(filename, 'w')
                
    
        file = h5py.File(filename, 'r+')
        file.create_dataset('Parameters', data = np.array(para))
        file.create_dataset('Delta', data = delta)
        file.close()
    
        
        final_sequences = np.zeros((alphas.shape[0],number_sequences,number_spins),dtype = np.int8)
    
        for idxa,alpha in enumerate(alphas):
    
            kappa = kappaprime/np.dot(delta,delta)
       
            matrix_chains = np.zeros([number_sequences,number_spins],dtype = np.int8)
            
            final_sequences[idxa,:,:] = Run(matrix_chains, number_accepted_flips,number_sequences,number_spins,delta,alpha,kappa)
            
            
        file = h5py.File(filename, 'r+')
        file.create_dataset('Chains', data = final_sequences, dtype = np.int8)
        file.create_dataset('Kappa', data = np.array(kappaprime))
        file.create_dataset('Alpha', data = np.array(alphas))
        file.create_dataset('NumberSequences', data = np.array(number_sequences))
        file.create_dataset('NumberSpins', data = np.array(number_spins))
        file.create_dataset('NumberFlips', data = np.array(number_accepted_flips))
        file.close()
    