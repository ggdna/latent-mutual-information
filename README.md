# Approximating mutual information of high-dimensional variables using learned representations

This repository provides code to reproduce results from **Approximating mutual information of high-dimensional variables using learned representations.**

## Requirements


**TODO** Add scanpy and pandas as requirements for preprocessing. And BMI library. And potentially everything else that gets imported in the jupyter notebooks.

To install requirements:
```
pip install -r requirements.txt
```
Some datasets (to reproduce hematopoiesis and pLM results) must be downloaded separately. Instructions below:

1. **Downloading protein embeddings from UniProt.** Download [A. thaliana](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/UP000006548_3702/per-protein.h5) PT5 mean-pooled embeddings and [E. coli](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/UP000000625_83333/per-protein.h5) PT5 mean-pooled embeddings using the UniProt links. Save them as `data/ProtT5_embeddings/athaliana_embeddings.h5` and `data/ProtT5_embeddings/ecoli_embeddings.h5` respectively.

2. **Downloading hematopoiesis LT-Seq data.** Download all files from Experiment 1 from the [Weinreb et al., 2020 data repository](https://github.com/AllonKleinLab/paper-data/tree/master/Lineage_tracing_on_transcriptional_landscapes_links_state_to_fate_during_differentiation). Save them in `data/ltseq/`.

## Preprocessing

To preprocess pLM data and hematopoiesis data, run the `data/*_preprocessing.ipynb` notebooks.

## Results

The results from the paper can be reproduced by running Jupyter notebooks in the `analysis/` folder. Below, we indicate which notebooks correspond to which result.

| Figure/table | Result description | Filename | Notes |
|--------------|--------------------|----------|-------|
| Figure 2     | Dimensionality scaling, multivariate Gaussians | `B_Gaussian_grid.ipynb` | |

## Visualizing results

Results from the `analysis` notebooks are saved as `.csv` in the `results/` folder. These are then "pretty-plotted" in Jupyter notebooks, with names ending in `*_plots.ipynb`. For arranging as figures in the paper, we sometimes have notebooks called `*_arrangement.ipynb`.

## Acknowledgements

Early Keras implementations of the AECross model were written in collaboration with Pippa Richter. 