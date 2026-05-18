# AI-Powered Applicant Tracking System
 
### Master Thesis — Hitik Sharma
**University of Passau | Lehrstuhl für Informatik und Mathematik**
**Supervisor: Prof. Dr. Tomas Sauer**
 
---
 
> *An intelligent resume screening system that combines Deep Learning, Natural Language Processing, Credibility Verification, and Explainable AI to automate and enhance the recruitment pipeline — from raw resume upload to transparent, explainable hiring decisions.*
 
---
 
## Key Results at a Glance
 
| Metric | Value |
|--------|-------|
| Best Classification Accuracy | **100.00%** (Linear SVM) |
| Deep Learning Accuracy | **98.97%** (BiLSTM + Attention) |
| Models Trained & Compared | **12** (6 TF-IDF + 4 SBERT + 2 Deep Learning) |
| Skill Taxonomy | **200+ skills** across **11 categories** |
| Resume Categories | **25 job categories** |
| Datasets Integrated | **8 diverse sources** (incl. O*NET from US Dept. of Labor) |
| Explainability Methods | **SHAP + LIME** (dual validation) |
| Credibility Gate | **6-layer** adversarial robustness filter |
| Thesis Figures | **40+** publication-quality visualizations |
 
---
 
## What This Project Does
 
Traditional Applicant Tracking Systems rely on simple keyword matching — they can't tell the difference between someone who *"managed a machine learning team"* and someone who just listed *"machine learning"* as a skill. This thesis builds a complete, research-grade ATS that:
 
1. **Classifies resumes** into 25 job categories using NLP and deep learning
2. **Matches candidates to jobs** using both skill overlap and semantic similarity (SBERT)
3. **Scores resume quality** through a multi-task neural network
4. **Verifies credibility** with automated completeness scoring, skill consistency checks, and red flag detection
5. **Explains every decision** using SHAP (game theory) and LIME (perturbation-based) — so recruiters know *why* a candidate was ranked
6. **Rejects non-resume inputs** through a 6-layer credibility gate that catches adversarial inputs (tested with Bible text, recipes, news articles, and cover letters)
---
 
## System Architecture
 
```
                    ┌─────────────────┐
                    │   PDF / DOCX    │
                    │   Resume Upload │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Text Extraction │
                    │  (PyMuPDF/docx2txt)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  7-Stage NLP    │
                    │  Preprocessing  │
                    │  Pipeline       │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼──────┐ ┌────▼─────┐ ┌──────▼───────┐
    │  TF-IDF (5000) │ │  SBERT   │ │ 14 Handcraft │
    │  Sparse Features│ │ (384-dim)│ │  Features    │
    └─────────┬──────┘ └────┬─────┘ └──────┬───────┘
              │              │              │
    ┌─────────▼──────┐ ┌────▼─────┐ ┌──────▼───────┐
    │  12 ML/DL      │ │ Resume-JD│ │ Multi-Task   │
    │  Classifiers   │ │ Matching │ │ Scorer       │
    └─────────┬──────┘ └────┬─────┘ └──────┬───────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼──────┐ ┌────▼─────┐ ┌──────▼───────┐
    │  6-Layer       │ │ SHAP +   │ │ Credibility  │
    │  Credibility   │ │ LIME     │ │ Scoring      │
    │  Gate          │ │ Explain  │ │ + Red Flags  │
    └─────────┬──────┘ └────┬─────┘ └──────┬───────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Streamlit Web  │
                    │  Application    │
                    └─────────────────┘
```
 
---
 
## Project Structure
 
```
Thesis/
│
├── notebooks/                          # Core research pipeline
│   ├── 01_data_collection_and_setup.ipynb     # 8 dataset collection & project setup
│   ├── 02_exploratory_data_analysis.ipynb     # 17 EDA figures, statistical analysis
│   ├── 03_nlp_preprocessing_pipeline.ipynb    # 7-stage NLP + TF-IDF + SBERT
│   ├── 04_deep_learning_model_training.ipynb  # 12 models: SVM, RF, BiLSTM, Multi-Task
│   ├── 05_shap_explainability.ipynb           # SHAP + LIME dual explainability
│   ├── 06_credibility_verification.ipynb      # Completeness, consistency, red flags
│   ├── 07_ats_web_application.ipynb           # Streamlit app generation
│   └── 08_scientific_experiments.ipynb        # Adversarial testing & robustness
│
├── app/
│   └── app.py                             # Streamlit ATS web application (V2)
│
├── reports/
│   ├── figures/                           # 40+ publication-quality thesis figures
│   └── tables/                            # Dataset & model comparison tables
│
├── data/
│   └── processed/                         # Processed data, taxonomies, encoders
│
├── configs/                            # Project configuration
├── requirements.txt                    # Python dependencies
└── README.md                           # You are here
```
 
---
 
##  Quick Start
 
```bash
# Clone the repository
git clone https://github.com/hitiksharma/AI-Powered-ATS.git
cd AI-Powered-ATS
 
# Set up environment
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
 
# Run the web application
streamlit run app/app.py
```
 
Then open **http://localhost:8501** in your browser.
 
---
 
##  Models & Performance
 
### Classification Results (Test Set, n=97)
 
| Rank | Model | Features | Accuracy | F1-Score |
|------|-------|----------|----------|----------|
| 1 | **Linear SVM** | TF-IDF | **100.00%** | **1.0000** |
| 2 | **Gradient Boosting** | TF-IDF | **100.00%** | **1.0000** |
| 3 | BiLSTM + Attention | TF-IDF | 98.97% | 0.9896 |
| 4 | Multi-Task Scorer | Handcrafted | 98.97% | 0.9903 |
| 5 | SVM + SBERT | SBERT | 98.97% | 0.9900 |
| 6 | Random Forest | TF-IDF | 98.97% | 0.9898 |
 
### Why Linear SVM Outperforms Deep Learning Here
 
The 25 job categories have highly distinctive vocabularies — "java," "spring," "hibernate" appear almost exclusively in Java Developer resumes. TF-IDF bigram features capture these lexical signatures perfectly, making linear separation trivially achievable. Deep learning adds value through the Multi-Task Scorer's quality scoring capability and the BiLSTM's potential for generalization to noisier, real-world data.
 
---
 
## 6-Layer Credibility Gate
 
A key research contribution — addressing the **open-set recognition problem** where the classifier assigns predictions to ANY input, even non-resumes:
 
| Layer | Check | What It Catches |
|-------|-------|-----------------|
| 1 | Document Length ≥ 40 words | Empty/minimal uploads |
| 2 | Document Type Detection | Cover letters, news articles, recipes |
| 3 | Skill Extraction ≥ 3 skills | Generic text without professional skills |
| 4 | Classification Confidence ≥ 50% | Ambiguous or out-of-domain inputs |
| 5 | Resume Structure ≥ 2 sections | Unstructured documents |
| 6 | Category-Skill Consistency | Mismatched classifications |
 
**Adversarial Testing Results:**
 
| Input | Verdict | Confidence | Skills Found |
|-------|---------|------------|-------------|
| ✅ Data Science Resume | ACCEPT | 70.9% | 10 |
| ✅ Cyber Security Resume | ACCEPT | 60.5% | 9 |
| ❌ Bible (849K words) | ACCEPT WITH REVIEW* | 5.6% | 28 (false positives) |
| ❌ Cover Letter | REJECT | 8.8% | 3 |
| ❌ Recipe | REJECT | 6.7% | 0 |
| ❌ News Article | REJECT | 8.6% | 1 |
| ❌ Random Gibberish | REJECT | 7.5% | 0 |
 
*\*The Bible bypasses pattern-based detection due to its extreme length causing statistical false-positive matches — documented as a known limitation with proposed solutions in the thesis.*
 
---
 
##  Explainability (SHAP + LIME)
 
The system doesn't just predict — it **explains why**:
 
- **SHAP** (SHapley Additive exPlanations): Mathematically grounded in cooperative game theory. Shows which words push the prediction toward or away from each category.
- **LIME** (Local Interpretable Model-agnostic Explanations): Perturbs the input and fits a local linear model to explain individual predictions.
Both methods independently identify the same key features (e.g., "python," "developer," "machine learning" for Data Science resumes), providing **methodological triangulation** that validates the model's reasoning.
 
---
 
##  Tech Stack
 
| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| Deep Learning | PyTorch |
| Machine Learning | scikit-learn |
| NLP | spaCy, NLTK, Sentence-BERT |
| Explainability | SHAP, LIME |
| Feature Engineering | TF-IDF, SBERT (all-MiniLM-L6-v2) |
| Web Application | Streamlit |
| Data Processing | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Skills Database | O*NET (US Dept. of Labor) |
 
---
 
##  Datasets
 
| # | Dataset | Records | Purpose |
|---|---------|---------|---------|
| 1 | Resume Corpus | 962 | Primary labeled training data (25 categories) |
| 2 | 54K Structured Resumes | 54,933 | Large-scale skill & pattern analysis |
| 3 | Job Description Corpus | 50,000+ | Demand-side data for resume-JD matching |
| 4 | Resume-JD Pairs | ~5,000 | Supervised similarity training |
| 5 | Resume NER Dataset | 220 | Named entity recognition annotations |
| 6 | Career Skills Dataset | ~5,000 | Job-skill mapping for consistency checking |
| 7 | O*NET Database | 1,016 | Ground-truth occupational skill taxonomy |
| 8 | Custom Skills Taxonomy | 200+ | 11-category skill classification system |
 
---
 
##  Thesis
 
**Title:** AI-Powered Applicant Tracking System with Deep Learning, Credibility Verification, and Explainable AI
 
**Author:** Hitik Sharma
 
**Institution:** University of Passau, Lehrstuhl für Informatik und Mathematik
 
**Supervisor:** [Prof. Dr. Tomas Sauer](https://www.fim.uni-passau.de/digitale-bildverarbeitung)
 
**Year:** 2026
 
---
 
##  License
 
This project was developed as part of a Master's thesis at the University of Passau. For academic use and reference.
 
---
 
##  Acknowledgments
 
- Prof. Dr. Tomas Sauer for supervision and guidance
- University of Passau, Faculty of Computer Science and Mathematics
- O*NET Resource Center, US Department of Labor
- The open-source community behind PyTorch, scikit-learn, SHAP, LIME, and Streamlit
 
