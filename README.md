# AI-Powered Applicant Tracking System
### Master Thesis — Hitik Sharma, University of Passau, 2026

An intelligent resume screening system combining Deep Learning, NLP, Credibility Verification, and Explainable AI.

## Key Results
- **12 models trained** — best: Linear SVM (100%), BiLSTM+Attention (98.97%)
- **200+ skill taxonomy** across 11 categories
- **6-layer credibility gate** for adversarial robustness
- **SHAP + LIME** dual explainability
- **Streamlit web app** with PDF/DOCX upload
## Project Structure
Thesis_final/
├── notebooks/
│   ├── 01_data_collection_and_setup.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_nlp_preprocessing_pipeline.ipynb
│   ├── 04_deep_learning_model_training.ipynb
│   ├── 05_shap_explainability.ipynb
│   ├── 06_credibility_verification.ipynb
│   ├── 07_ats_web_application.ipynb
│   └── 08_scientific_experiments.ipynb
├── app/
│   └── app.py                    # Streamlit ATS application
├── reports/
│   └── figures/                  # All 40+ thesis figures
├── thesis_template/              # LaTeX thesis source
└── data/
└── processed/                # Processed data files

## How to Run
```bash
git clone https://github.com/hitiksharma/AI-Powered-ATS.git
cd AI-Powered-ATS
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Tech Stack
Python | PyTorch | scikit-learn | SBERT | SHAP | LIME | Streamlit | spaCy | NLTK

## Thesis
Supervised by Prof. Dr. Michael Granitzer, Lehrstuhl für Data Science, University of Passau.
