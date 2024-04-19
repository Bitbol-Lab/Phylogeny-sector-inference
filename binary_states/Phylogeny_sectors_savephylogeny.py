#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
save phylogeny of sequences
"""
import sys
import os
import numpy as np
import random
import time

# set seed
seed = 12
random.seed(seed)

from tqdm import tqdm
import matplotlib.pyplot as plt
from math import exp, expm1
from datetime import datetime, date
import h5py
startTime = datetime.now()
today = date.today()
from numba import jit

def OpenTempFile(path,paramkey,paramvalue,idxc):
    file = h5py.File(path,'r')
    sequences = np.array(file['Chains'])
    parameters= np.array(file[paramkey])
    idx = np.where(parameters == paramvalue)
    file.close()
    return sequences[idx,idxc,:][0,0,:].copy()
        
def GenerateDeltaNormal(average_sector,average_nonsector, variance_sector,variance_nonsector, number_spins, number_components_sector):
    
    delta = np.zeros(number_spins)
    
    delta[0:number_components_sector] = np.random.normal(average_sector,variance_sector,number_components_sector)
    delta[number_components_sector:number_spins] = np.random.normal(average_nonsector,variance_nonsector,number_spins - number_components_sector)
    
    return delta

@jit(nopython=True, fastmath = True)
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

def Generate_tree(nbr_gen):
    tree = {}
    for g in range(0,nbr_gen+1):
        list_children = np.linspace(1,pow(2,g),pow(2,g), dtype = np.int16)
        for child in list(list_children):
            tree['{}/{}'.format(g,child)] = None
    return tree

def Run1Tree(number_generations,number_spins,delta, alpha, kappa,number_mutations,indexc,bl_eqstart,kappaprime,paramkey,paramvalue,pathtoEQ):

    Tree = Generate_tree(number_generations)
    all_chains = np.zeros((2*pow(2,number_generations)-1,number_spins)) 

    starting_chain = OpenTempFile(pathtoEQ,paramkey,paramvalue,indexc)

    all_chains[0,:] = starting_chain.copy()

    Tree['0/1'] = starting_chain

    ct = 0
    for g in range(1,number_generations+1):
        
        list_parents = np.linspace(1,pow(2,g-1),pow(2,g-1), dtype = np.int16)

        for parent in list_parents:

            chain = Tree['{}/{}'.format(g-1,parent)]
            
            newchain1 = MCEvolution(chain.copy(), delta, alpha, kappa,number_mutations, number_spins)
            newchain2 = MCEvolution(chain.copy(),delta, alpha, kappa,number_mutations, number_spins)

            Tree['{}/{}'.format(g,2*parent-1)] = newchain1
            Tree['{}/{}'.format(g,2*parent)] = newchain2
            
            all_chains[ct+1,:] = newchain1
            all_chains[ct+2,:] = newchain2
            ct = ct+2
    final_chains = np.zeros([pow(2,number_generations),number_spins],dtype = np.int8)
    
    for index_chain, child in enumerate(list(np.linspace(1,pow(2,number_generations),pow(2,number_generations),dtype = np.int16))):
        final_chains[index_chain,:] = Tree['{}/{}'.format(number_generations,child)]
    
    return final_chains,all_chains
    
if __name__ == '__main__':
        
    number_spins = 200                 
    number_accepted_flips = 3000
    
    average_sector = 5; variance_sector = 0.25
    average_nonsector = 0.5; variance_nonsector = 0.25
    number_components_sector = 20
    
    # delta = GenerateDeltaNormal(average_sector,average_nonsector, variance_sector,variance_nonsector, number_spins, number_components_sector)
    delta = np.load('./vector_mut_effects_2states/delta.npy')
    
    pathtoEQfile = './example_generated_data/EQ_K10_alpham300_300_10real/2024_04_19_10_49_30_Sectors_nspins200_flips3000_nseq2048_seed_34_Alphamin-300_Alphamax300_K10_realnbr0.h5'
    idxs_chainseq = np.linspace(0,2047,2048)
    np.random.shuffle(idxs_chainseq)
    paramkeyeq = 'Alpha'
    
    alpha = 90
    number_realisations = 10
    number_mutations = 50
    
    for real in tqdm(range(0,number_realisations)):
            indexc = 0
            number_generations = 11
        
            kappaprime = 10
            kappa = kappaprime/np.dot(delta,delta)  
            
            date = today.strftime("%Y_%m_%d_")
            hour = startTime.strftime('%H_%M_%S_')
            
    
            path = './example_generated_data/phylogeny_Alpha90_K40_M50_savephylogeny/'
    
            if not os.path.exists(path):
                os.makedirs(path)
                
            filename = path+date+hour+'Sectors_nspins{}_flips{}_seed_{}_deltanormal_M{}_Kappa{}_G{}_realnbr{}.h5'.format(number_spins, number_accepted_flips,seed,int(number_mutations),int(kappaprime),number_generations,real)
            file = h5py.File(filename, 'w')
            file.close()
            
            para = [number_spins, number_accepted_flips,number_components_sector, average_sector, variance_sector,average_nonsector,variance_nonsector, seed,number_mutations,kappaprime,number_generations]
            
            file.create_dataset('Parameters', data = np.array(para))
            file.create_dataset('Delta', data = delta)
            file.close()
            
            finalchains = np.zeros((pow(2,number_generations),number_spins))
    
            chains_lastg,allchains = Run1Tree(number_generations,number_spins,delta, alpha, kappa,number_mutations,int(idxs_chainseq[indexc]),kappaprime,paramkeyeq,alpha,pathtoEQfile)
    
            finalchains[:,:] = chains_lastg.copy()
        
            file = h5py.File(filename, 'r+')
    
            file.create_dataset('Chains', data = finalchains, dtype = np.int8)
            file.create_dataset('allchains', data = allchains, compression='gzip', compression_opts=9)
            file.create_dataset('Kappa', data = np.array(kappaprime))
            file.create_dataset('Alpha', data = np.array(alpha))
            file.create_dataset('Mutations', data = np.array(number_mutations))
            file.create_dataset('NumberGenerations', data = np.array(number_generations))
                    
            file.create_dataset('NumberSequences', data = np.array(pow(2,number_generations)))
            file.create_dataset('NumberSpins', data = np.array(number_spins))
            file.create_dataset('NumberFlips', data = np.array(number_accepted_flips))
            file.close()
            
    
