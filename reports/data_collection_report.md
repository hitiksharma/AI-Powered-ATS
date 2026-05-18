
# Data Collection Report
# Master Thesis: AI-Powered ATS with Deep Learning & Explainable AI
# Generated: 2026-03-17 16:34:12

## Datasets Collected

| # | Dataset | Records | Source | Purpose |
|---|---------|---------|--------|---------|
| 1 | Resume Corpus | 2,400+ | Kaggle (snehaanbhawal) | Resume text classification |
| 2 | Updated Resume Dataset | Extended | Kaggle (jillanisofttech) | Expanded training data |
| 3 | Job Descriptions | 30,000+ | Kaggle (ravindrasinghrana) | JD matching & analysis |
| 4 | LinkedIn Jobs & Skills | 1.3M+ | Kaggle (asaniczka) | Skill taxonomy & patterns |
| 5 | Resume-JD Pairs | Matched | Kaggle (pranavvenugo) | Supervised similarity training |
| 6 | Resume NER | Annotated | Kaggle (dataturks) | Named entity recognition |
| 7 | Job Skill Set | Mapped | Kaggle (batuhanmutlu) | Skill-to-job mapping |
| 8 | O*NET Database | 1,016 occupations | US Dept. of Labor | Gold-standard skill taxonomy |
| 9 | 54K Structured Resumes | 54,000 | Kaggle (suriyaganesh) | Large-scale training |
| 10 | Custom Skills Taxonomy | 206 skills | Custom curated | Skill extraction & validation |

## Scientific Rationale

Our multi-source data strategy addresses key limitations in prior work:
- Single-dataset studies (Papers 1, 8) suffer from limited generalizability
- SATYA (Paper 5) used only 1,000 manually scored resumes
- Our approach combines 50,000+ resumes + 30,000+ JDs + O*NET taxonomy
- LinkedIn data provides real-world skill distribution patterns
- O*NET provides institutional credibility and ground-truth validation
