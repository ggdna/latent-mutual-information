# Approximating mutual information of high-dimensional variables using learned representations

This repository provides code to reproduce results from **Approximating mutual information of high-dimensional variables using learned representations.**

*author list redacted for peer review*

## Requirements


**TODO** Add scanpy and pandas as requirements for preprocessing. And BMI library. And potentially everything else that gets imported in the jupyter notebooks.

To install requirements:
```
pip install -r requirements.txt
```
Some datasets (to reproduce hematopoiesis and pLM results) must be downloaded separately. Instructions below:

1. **Downloading protein embeddings from UniProt.** Download [A. thaliana](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/UP000006548_3702/per-protein.h5), [H. sapiens](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/UP000005640_9606/per-protein.h5) and [E. coli](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/UP000000625_83333/per-protein.h5) PT5 mean-pooled embeddings using the UniProt links. Save them as `data/ProtT5_embeddings/athaliana_embeddings.h5`, `data/ProtT5_embeddings/hsapiens_embeddings.h5` and `data/ProtT5_embeddings/ecoli_embeddings.h5` respectively.

2. **Downloading hematopoiesis LT-Seq data.** Download all files from Experiment 1 from the [Weinreb et al., 2020 data repository](https://github.com/AllonKleinLab/paper-data/tree/master/Lineage_tracing_on_transcriptional_landscapes_links_state_to_fate_during_differentiation). Save them in `data/ltseq/`.

## Preprocessing

To preprocess pLM data and hematopoiesis data, run the `data/*_preprocessing.ipynb` notebooks.

## Results

The results from the paper can be reproduced by running Jupyter notebooks in the `analysis/` folder. Below, we indicate which notebooks correspond to which result.

| Figure/table | Result description | Filename | Notes |
|--------------|--------------------|----------|-------|
| Figure 2     | Dimensionality scaling, multivariate Gaussians | `B_Gaussian_grid.ipynb` | includes alternate regularizers |
| Figure 3 | Convergence rates | `B_Gaussian_sample_complexity.ipynb`| |
| Figure 4 | Benchmarking by resampling (MNIST) | `B_MNIST.ipynb` | |
|Figure 4 | Benchmarking by resampling (PT5) | `B_species_mixing.ipynb`| |
| Figure 5 | Ligand-Receptor MI| `OP_LR_shuffle_test.ipynb`| |
| Figure 5 | Kinase-target MI| `OP_kinase_shuffle_test.ipynb`| |
| Figure 5 | Ligand-Receptor prediction| `OP_LR_classification.ipynb`| |
| Figure 5 | Kinase-target prediction| `OP_KT_classification.ipynb`| |
| Figure 6 | Hematopoiesis markov test | `H_markov_test.ipynb`| |
| Figure 6 | Hematopoiesis pMI decomposition | `H_pseudotime_pMI.ipynb`| |
| Appendix | Validating MNIST assumptions | `V_MNIST_assumptions.ipynb` | |
| Appendix | Validating ProtTrans5 assumptions | `V_PT5_assumptions.ipynb` | |
| Appendix | Comparing latent estimators | `A_comparing_latent_estimators.ipynb` | |
| Appendix | Interpreting MI by dissecting decoders | `A_decoder_inspection.ipynb` | |



## Visualizing results

Results from the `analysis` notebooks are saved as `.csv` in the `results/` folder. These are then "pretty-plotted" in Jupyter notebooks with names ending in `*_plots.ipynb` or `*_arrangement.ipynb` for organizing the more complicated figures in the paper.

## Acknowledgements

*redacted for peer review*