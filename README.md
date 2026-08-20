# xpdm — Explainable Predictive Maintenance

**Project:** Why Did the Model Flag This Machine?  
Explainable Deep Learning for Predictive Maintenance using NASA CMAPSS datasets.

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Dataset Download
1. Go to: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
2. Download **CMAPSS** (Turbofan Engine Degradation Simulation)
3. Place the 14 `.txt` files into `data/raw/CMAPSS/`

Expected files:
  - train_FD001.txt, test_FD001.txt, RUL_FD001.txt
  - train_FD002.txt, test_FD002.txt, RUL_FD002.txt
  - train_FD003.txt, test_FD003.txt, RUL_FD003.txt
  - train_FD004.txt, test_FD004.txt, RUL_FD004.txt

## Project Structure
```
xpdm/
├── data/
│   ├── raw/CMAPSS/        ← put downloaded .txt files here
│   └── processed/         ← preprocessed arrays saved here later
├── notebooks/             ← Jupyter notebooks (EDA, training, XAI)
├── src/
│   ├── data/              ← data loading and preprocessing scripts
│   └── utils/             ← config and shared utilities
└── outputs/
    ├── figures/           ← saved plots
    └── results/           ← saved metrics and results
```
