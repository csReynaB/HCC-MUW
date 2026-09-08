# Deciphering the microbiota–immune axis in liver health and disease

This repository contains the processed data, analysis code, and reproducible workflows supporting the manuscript:

> **Deciphering the microbiota–immune axis in liver health and disease yields diagnostic and prognostic serological signatures**
>
> Carlos S. Reyna-Blanco, Bernhard Scheiner, *et al.* (manuscript in revision)

The study uses phage immunoprecipitation sequencing (**PhIP-seq**) to characterize antibody–antigen interactions across liver-healthy controls, patients with cirrhosis, and patients with hepatocellular carcinoma (**HCC**). The repository follows the main analytical flow of the study: describing the serological landscape, evaluating diagnostic signatures, and assessing their prognostic relevance.

## Repository overview

| Directory | Content |
| --- | --- |
| [`descriptive/`](descriptive/) | R workflow for cohort characteristics, antibody-repertoire summaries, alpha and beta diversity, and peptide- and taxon-level comparisons across disease groups, etiologies, treatments, and response categories. |
| [`classification/`](classification/) | Python workflow for diagnostic classification using `phipml`, including nested cross-validation, Random Forest and XGBoost models, model evaluation, and SHAP-based interpretation. Configurations are provided for Controls vs HCC, Controls vs Cirrhosis, Cirrhosis vs HCC, and Controls vs liver disease. |
| [`survival/`](survival/) | Python workflow for prognostic modeling using `phipsurv`, with XGBoost–Cox models, nested cross-validation, time-dependent performance metrics, Kaplan–Meier analyses, and SHAP-based feature interpretation in the ICI- and TKI-treated HCC cohorts. |

Each analysis directory contains its corresponding input enriched presence/absence peptide data, metadata, source code, configuration files, and notebooks. The included `exist.csv` files are processed binary peptide-enrichment matrices; they are not raw sequencing reads.

## Reproducing the analyses

Clone the repository:

```bash
git clone https://github.com/csReynaB/HCC-MUW.git
cd HCC-MUW
```

The three components use separate workflows:

- **Descriptive analyses:** open [`descriptive/notebook/Desciptive_analyisis_alldata.Rmd`](descriptive/notebook/Desciptive_analyisis_alldata.Rmd) in RStudio and render the notebook after installing the packages loaded at the beginning of the document.
- **Diagnostic classification:** follow the installation and execution instructions in the [`classification` README](classification/README.md). Manuscript-specific main comparisons are defined in [`classification/configs/`](classification/configs/).
- **Prognostic analysis:** follow the [`survival` README](survival/README.md). The main manuscript analyses are documented in the notebooks for the [`HCC–ICI`](survival/notebooks/survivalAnalysis_HCC-ICI_fig5.ipynb) and [`HCC–TKI`](survival/notebooks/survivalAnalysis_HCC_TKI_fig5.ipynb) cohorts.

Reproducible Python environments are specified in the `ML_env.yml` and `pyproject.toml` files within the classification and survival directories. Dockerfiles and example shell scripts are also provided for containerized or HPC execution.

## Data organization

Within each analysis module:

- `Data/` contains processed peptide-enrichment matrices.
- `Metadata/` contains anonymized sample information and peptide-library annotations used by the workflows.
- `configs/` contains manuscript-specific analysis settings.
- `notebooks/` contains the analyses used to assemble the corresponding manuscript figures.
- `src/` contains the reusable Python or R analysis functions.

## Citation

If you use the code or data in this repository, please cite the archived release:

> Reyna-Blanco, C. S. (2026). *csReynaB/HCC-MUW: HCC-MUW* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.22656008

Please also cite the associated manuscript. Its complete journal citation will be added once the article is published.

Software-specific citation metadata for `phipml` and `phipsurv` are provided in the corresponding `CITATION.cff` files.

## License

The classification and survival code is distributed under the MIT License; see the `LICENSE` file within each directory. Any reuse of the accompanying data should also follow the terms of the associated manuscript and its data-availability statement.

## Contact

For questions about the analysis code, please contact **Carlos S. Reyna-Blanco** or open an issue in this repository.
