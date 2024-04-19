#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure phylogeny, no selection
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


@jit(nopython=True)
def Evolution(sequence,number_accepted_flips, number_spins):

    counter = 0
    while(counter < number_accepted_flips):

        rand_site = random.randrange(0,number_spins,1)

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

def Run1Tree(number_generations,number_spins,number_mutations):

    
    Tree = Generate_tree(number_generations)
    

    starting_chain = 2*(np.random.randint(0,2,number_spins))-1
   
    Tree['0/1'] = starting_chain
    

    for g in range(1,number_generations+1):
        
        list_parents = np.linspace(1,pow(2,g-1),pow(2,g-1), dtype = np.int16)

        for parent in list_parents:

            chain = Tree['{}/{}'.format(g-1,parent)]
            
            newchain1 = Evolution(chain.copy(),number_mutations, number_spins)
            newchain2 = Evolution(chain.copy(),number_mutations, number_spins)
   
            Tree['{}/{}'.format(g,2*parent-1)] = newchain1
            Tree['{}/{}'.format(g,2*parent)] = newchain2
            
    final_chains = np.zeros([pow(2,number_generations),number_spins],dtype = np.int8)
    
    for index_chain, child in enumerate(list(np.linspace(1,pow(2,number_generations),pow(2,number_generations),dtype = np.int16))):
        final_chains[index_chain,:] = Tree['{}/{}'.format(number_generations,child)]
    
    return final_chains
    

number_spins = 200

number_realisations = 1
number_mutations = np.linspace(1,50,50)
for real in tqdm(range(0,number_realisations)):
        number_generations = 11
        date = today.strftime("%Y_%m_%d_")
        hour = startTime.strftime('%H_%M_%S_')
        
        path = './example_generated_data/pure_phylogeny/'

        if not os.path.exists(path):
            os.makedirs(path)
        
        filename = path+date+hour+'Sectors_nspins{}_seed_{}_G{}_realnbr{}.h5'.format(number_spins,seed,number_generations,real)
        file = h5py.File(filename, 'w')
        
        
        para = [number_spins,seed,number_generations]
        
        file.create_dataset('Parameters', data = np.array(para))
        file.close()
        
        finalchains = np.zeros((number_mutations.shape[0],pow(2,number_generations),number_spins))
        
        for idxm,nbrmutations in enumerate(number_mutations):
            chains_lastg = Run1Tree(number_generations,number_spins,nbrmutations)

            finalchains[idxm,:,:] = chains_lastg.copy()
        
        file = h5py.File(filename, 'r+')
        file.create_dataset('Chains', data = finalchains, dtype = np.int8)
        file.create_dataset('Mutations', data = np.array(number_mutations))
        file.create_dataset('NumberGenerations', data = np.array(number_generations))
        file.create_dataset('NumberSequences', data = np.array(pow(2,number_generations)))
        file.create_dataset('NumberSpins', data = np.array(number_spins))
        file.close()
        
   