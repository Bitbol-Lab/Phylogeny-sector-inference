# Phylogeny-sector-inference

## Getting started

Clone this repository on your local machine by running
```bash
git clone git@github.com:Bitbol-Lab/Phylogeny-sector-inference.git
```
and move inside the root folder.

### 2 states
The binary_states folder contains four scripts to generate sequences. First, ```Sectors_equilibrium.py``` allows to generate independent sequences at equilibrium. Second, ```Phylogeny_sectors.py``` allows to generate sequences with phylogeny. Note that an equilibrium dataset is needed to generate sequences with phylogeny because an equilibrium sequence is taken as the root of the phylogeny. 
Third, ```Phylogeny_sectors_savephylogeny.py``` allows to generate sequences with phylogeny and save the whole phylogeny. Finally, ```PurePhylogenyNoselection_sectors.py``` allows to generate sequences with only phylogeny and no selection. 

The ```Analyze_sectors_binarydata.py``` allows to infer sectors using ICOD, covariance, SCA or conservation on the example data stored in the "example_generated_data" folder. The same script reproduces the figures with the same results as in the article. The results for the article figures are stored in the "results" folder. The necessary functions are in the main and can be commented if not needed. For SCA, we used the the ```pySCA``` github package (https://github.com/reynoldsk/pySCA).

### Natural data
The "natural_data" folder contains two excel sheets ("41592_2018_138_MOESM4_ESM(1).xls" and "Metadata_dms_keysheetnames.xls") that contain the information on the DMS and on the reference sequences used to construct the MSA. These files comes from _Adam J. Riesselman, John B. Ingraham, and Debora S. Marks. Deep generative models of genetic
variation capture the effects of mutations. Nature Methods, 15(10):816–822, Oct 2018_. 

First, ```Riesselman_dataset_buildRawMSA_DMS.py``` needs to be run in order to construct raw MSAs and the DMS files, it is required to give the path to the uniref100.fasta file. Then, ```PrepareMSA_Infer_natdata.py``` allows to build MSAs at different cutoffs and save them. This script also allows to infer sectors using ICOD, MI, SCA and conservation, it saves the eigenvectors for each method and for each MSA. For SCA, we used the the ```pySCA``` github package (https://github.com/reynoldsk/pySCA).

Finally, ```Analyse_sectors_naturaldata.py``` script computes the AUC by taking eigenvectors and the corresponding DMS files which are fitted. The same script also produces the figures corresponding to natural data. It also computes the number of sector sites recovered by ICOD and conservation. 

Folders "conservation_sum" and "sectors_icod" contain, for each family, the summed conservation or the summed icod eigenvectors which allow to compute the number of sector sites found by ICOD and conservation. The "MSAs" folder contains one full example for the "Translation initiation factor IF1" family, meaning: every MSAs, each eigenvector for every method, DMS file, etc. The other families which are in the folder contain just their DMS file necessary to reproduce DMS histograms figure in the article. Finally, the "results" folder contains every AUC to reproduce the corresponding figures in the article.


