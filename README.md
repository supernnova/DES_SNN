# DES_SuperNNova
Application of SuperNNova to DES

- Visualization of data
original_data_visualization.py
- Testing light-curve skimming options
run_skim_and_classification.py
(Just the skimming can be done using skim_data_lcs.py)

- notebooks with data and peak exploration

Beware!!!
Skimmed photometry fits tables can't be used as input for SNANA fits
the original fits tables have assitional extensions with survey info

#Requirements:
## Clone SuperNNova
it must be cloned in this repository
```bash
git clone https://github.com/supernnova/supernnova.git
```

## Set up SuperNNova environment with conda

    cd SuperNNova/env

    # Create conda environment
    conda create --name <env> --file <conda_file_of_your_choice>

    # Activate conda environment
    source activate <env>