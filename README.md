# Exdyn
Exdyn is intended to analyze colocalization relation ships between single cell transcriptomes, integrating them with spatial transcriptome.

## Instalation
You can install exdyn using pip command from your shell.
```shell
pip install exdyn
```

## Usage
You need to prepare [`AnnData` objects](https://anndata.readthedocs.io/en/latest/) which includes spliced and unspliced transcript counts for single cell transcriptome respectively. You can see the usage in [IPython Notebook1](tutorial/sim_analysis_batched_experimental_conditions.ipynb) and [IPython Notebook2](tutorial/scc_analysis_colocalization_conditions.ipynb).