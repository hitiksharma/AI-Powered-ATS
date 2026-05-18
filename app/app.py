#!/usr/bin/env python3
"""
AI-Powered Applicant Tracking System (ATS) — Version 2
Master Thesis — Hitik Sharma, 2026

UPDATES in V2:
- 6-Layer Enhanced Credibility Gate
- Cover Letter Detection
- Document Type Classification
- Adversarial Input Rejection
- Improved skill-category consistency checking

Run: streamlit run app.py
"""

import os
import json
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter

# ---- Page Config ----
st.set_page_config(page_title="AI-Powered ATS", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")

# ---- Paths ----
PROJECT_ROOT = os.path.expanduser("~/Desktop/thesis_final")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data/processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ============================================================
# DOCUMENT TYPE DETECTOR
# ============================================================


def detect_document_type(text):
    text_lower = text.lower()

    # Cover letter indicators
    cover_letter_signals = 0
    cover_letter_patterns = [
        r'\bdear\s+(hiring|sir|madam|team|recruiter|manager)',
        r'\bi\s+am\s+writing\s+to\s+(express|apply|submit)',
        r'\b(sincerely|regards|respectfully|thank you for (your|considering))',
        r'\bi\s+believe\s+i\s+(would|could|can|am)',
        r'\b(position|role|opportunity)\s+(at|with|in)\s+your\s+(company|organization|firm)',
        r'\bi\s+am\s+(excited|eager|passionate|interested)\s+(about|to|in)',
        r'\bplease\s+(find|see)\s+(attached|enclosed|my)',
        r'\blook\s+forward\s+to\s+hearing',
    ]
    for pattern in cover_letter_patterns:
        if re.search(pattern, text_lower):
            cover_letter_signals += 1

    # Resume indicators
    resume_signals = 0
    sections_found = 0
    resume_section_headers = [
        r'\b(education|academic|qualification|degree)',
        r'\b(experience|employment|work\s+history|career)',
        r'\b(skills|technical|competencies|proficiency|expertise)',
        r'\b(projects?|portfolio|capstone)',
        r'\b(summary|objective|profile|about\s+me)',
        r'\b(certifications?|licenses?|awards?|achievements?)',
    ]
    for pattern in resume_section_headers:
        if re.search(pattern, text_lower):
            sections_found += 1
            resume_signals += 1

    resume_patterns = [
        r'\b\d+\+?\s*(years?|yrs?)\s*(of)?\s*(experience|exp)',
        r'\b(bachelor|master|ph\.?d|b\.?tech|m\.?tech|mba|b\.?sc)',
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        r'linkedin\.com/in/',
        r'github\.com/',
    ]
    for pattern in resume_patterns:
        if re.search(pattern, text_lower):
            resume_signals += 1

    if cover_letter_signals >= 2:
        return 'cover_letter', cover_letter_signals, sections_found
    elif resume_signals >= 3 and sections_found >= 2:
        return 'resume', resume_signals, sections_found
    elif resume_signals >= 2:
        return 'possible_resume', resume_signals, sections_found
    else:
        return 'other', resume_signals, sections_found


# ============================================================
# 6-LAYER CREDIBILITY GATE
# ============================================================
def credibility_check(text, tfidf_model, classifier, taxonomy, label_encoder,
                      confidence_threshold=50, min_skills=3, min_words=40):
    result = {
        'text_length': len(text.split()),
        'flags': [],
        'layers_passed': 0,
        'total_layers': 6,
    }

    # Layer 1: Document Length
    word_count = len(text.split())
    if word_count < min_words:
        result['flags'].append(('HIGH', 'INSUFFICIENT_LENGTH',
                               f'Document has only {word_count} words (minimum: {min_words})'))
    else:
        result['layers_passed'] += 1

    # Layer 2: Document Type Detection
    doc_type, doc_signals, sections_found = detect_document_type(text)
    result['document_type'] = doc_type
    result['sections_detected'] = sections_found

    if doc_type == 'cover_letter':
        result['flags'].append(
            ('HIGH', 'COVER_LETTER', f'This appears to be a cover letter, not a resume ({doc_signals} indicators)'))
    elif doc_type == 'other':
        result['flags'].append(
            ('HIGH', 'NOT_A_RESUME', f'This does not appear to be a resume ({doc_signals} resume indicators found)'))
    else:
        result['layers_passed'] += 1

    # Layer 3: Skill Extraction
    text_lower = text.lower()
    skills_found = []
    for category, skills in taxonomy.items():
        for skill in skills:
            if len(skill) <= 3:
                if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                    skills_found.append((skill, category))
            elif skill.lower() in text_lower:
                skills_found.append((skill, category))

    unique_skills = list({s[0]: s for s in skills_found}.values())
    result['skills_count'] = len(unique_skills)
    result['skills_found'] = unique_skills[:15]

    if len(unique_skills) < min_skills:
        result['flags'].append(
            ('MEDIUM', 'FEW_SKILLS', f'Only {len(unique_skills)} professional skills detected (minimum: {min_skills})'))
    else:
        result['layers_passed'] += 1

    # Layer 4: Classification Confidence
    X_clf = tfidf_model.transform([text])
    pred = classifier.predict(X_clf)[0]
    try:
        proba = classifier.predict_proba(X_clf)[0]
        confidence = max(proba) * 100
    except AttributeError:
        # LinearSVC doesn't have predict_proba — use decision_function instead
        decision = classifier.decision_function(X_clf)[0]
        # Convert decision scores to pseudo-probabilities using softmax
        exp_scores = np.exp(decision - np.max(decision))
        proba = exp_scores / exp_scores.sum()
        confidence = max(proba) * 100
    category = label_encoder.inverse_transform([pred])[0]

    result['predicted_category'] = category
    result['confidence'] = confidence

    top3_idx = np.argsort(proba)[-3:][::-1]
    result['top3'] = [(label_encoder.classes_[i], round(
        proba[i]*100, 1)) for i in top3_idx]

    if confidence < confidence_threshold:
        result['flags'].append(
            ('MEDIUM', 'LOW_CONFIDENCE', f'Classification confidence: {confidence:.1f}% (threshold: {confidence_threshold}%)'))
    else:
        result['layers_passed'] += 1

    # Layer 5: Resume Structure
    if sections_found < 2:
        result['flags'].append(
            ('MEDIUM', 'NO_STRUCTURE', f'Only {sections_found} resume sections detected (minimum: 2)'))
    else:
        result['layers_passed'] += 1

    # Layer 6: Category-Skill Consistency
    category_keywords = {
        'Data Science': ['python', 'machine learning', 'data', 'tensorflow', 'statistics', 'deep learning', 'pandas'],
        'Java Developer': ['java', 'spring', 'hibernate', 'maven', 'servlet', 'jsp'],
        'Python Developer': ['python', 'django', 'flask', 'pandas', 'numpy'],
        'Web Designing': ['html', 'css', 'javascript', 'react', 'angular', 'figma'],
        'HR': ['recruitment', 'payroll', 'onboarding', 'employee', 'hiring'],
        'DevOps Engineer': ['docker', 'kubernetes', 'aws', 'ci/cd', 'terraform', 'jenkins'],
        'Network Security Engineer': ['firewall', 'cisco', 'security', 'vpn', 'siem', 'penetration'],
        'Testing': ['selenium', 'testing', 'qa', 'automation', 'junit'],
        'Database': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'database'],
        'Blockchain': ['blockchain', 'ethereum', 'solidity', 'smart contract', 'web3'],
    }

    expected = category_keywords.get(category, [])
    if expected:
        matches = sum(1 for kw in expected if kw in text_lower)
        result['keyword_matches'] = matches
        if matches == 0 and confidence < 70:
            result['flags'].append(
                ('HIGH', 'CATEGORY_MISMATCH', f'No keywords for "{category}" found in document'))
        else:
            result['layers_passed'] += 1
    else:
        result['layers_passed'] += 1

# Final Verdict
    result['credibility_score'] = round(
        (result['layers_passed'] / result['total_layers']) * 100, 1)
    high_flags = sum(1 for f in result['flags'] if f[0] == 'HIGH')
    confidence_failed = any(f[1] == 'LOW_CONFIDENCE' for f in result['flags'])
    is_real_resume = result.get('document_type') in [
        'resume', 'possible_resume']

    if confidence_failed and result['confidence'] < 20 and not is_real_resume:
        result['verdict'] = 'REJECTED'
        result['verdict_color'] = 'red'
        result['verdict_msg'] = 'Document does not match any resume category and is not a professional resume.'
    elif confidence_failed and is_real_resume and result['layers_passed'] >= 4:
        result['verdict'] = 'ACCEPT WITH REVIEW'
        result['verdict_color'] = 'blue'
        result['verdict_msg'] = 'Resume accepted. Low confidence suggests candidate spans multiple domains — manual category assignment recommended.'
    elif result['layers_passed'] >= 5 and high_flags == 0:
        result['verdict'] = 'ACCEPT'
        result['verdict_color'] = 'green'
        result['verdict_msg'] = 'Document passes all credibility checks. Classification is reliable.'
    elif result['layers_passed'] >= 4 and high_flags == 0:
        result['verdict'] = 'ACCEPT WITH REVIEW'
        result['verdict_color'] = 'blue'
        result['verdict_msg'] = 'Document passes most checks. Classification provided with advisory.'
    elif result['layers_passed'] >= 3 and high_flags <= 1:
        result['verdict'] = 'MANUAL REVIEW'
        result['verdict_color'] = 'orange'
        result['verdict_msg'] = 'Multiple concerns detected. Human review recommended.'
    else:
        result['verdict'] = 'REJECTED'
        result['verdict_color'] = 'red'
        result['verdict_msg'] = 'Document does not meet resume requirements. Classification withheld.'

    return result


# ============================================================
# FILE READER
# ============================================================
def read_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return ""
    file_name = uploaded_file.name.lower()

    if file_name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8', errors='ignore')
    elif file_name.endswith('.pdf'):
        try:
            import fitz
            pdf_bytes = uploaded_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except ImportError:
            st.error("Install PyMuPDF: pip install PyMuPDF")
            return ""
    elif file_name.endswith('.docx'):
        try:
            import docx2txt
            from io import BytesIO
            return docx2txt.process(BytesIO(uploaded_file.read()))
        except ImportError:
            st.error("Install docx2txt: pip install docx2txt")
            return ""
    else:
        st.error(f"Unsupported: {file_name}. Use PDF, DOCX, or TXT.")
        return ""


def show_resume_uploader(key_prefix=""):
    uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"],
                                     key=f"{key_prefix}_upload", help="Drag and drop your file here")
    with st.expander("📝 Or paste text manually"):
        pasted_text = st.text_area("Paste text:", height=200, key=f"{key_prefix}_text",
                                   placeholder="Paste resume text here...")

    resume_text = ""
    if uploaded_file is not None:
        with st.spinner("📖 Reading file..."):
            resume_text = read_uploaded_file(uploaded_file)
            if resume_text.strip():
                st.success(
                    f"✅ Extracted {len(resume_text.split())} words from {uploaded_file.name}")
    elif pasted_text.strip():
        resume_text = pasted_text
    return resume_text


# ============================================================
# NLP FUNCTIONS
# ============================================================
def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\;\:\-\/\+\#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_contact(text):
    info = {}
    emails = re.findall(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    info['email'] = emails[0] if emails else None
    phones = re.findall(r"[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{7,15}", text)
    phones = [p.strip() for p in phones if len(re.sub(r'\D', '', p)) >= 10]
    info['phone'] = phones[0] if phones else None
    linkedin = re.findall(
        r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+", text)
    info['linkedin'] = linkedin[0] if linkedin else None
    github = re.findall(r"(?:https?://)?(?:www\.)?github\.com/[\w-]+", text)
    info['github'] = github[0] if github else None
    return info


def extract_education(text):
    text_lower = text.lower()
    for level, pattern in [("PhD", r"\b(ph\.?d|doctorate)\b"), ("Masters", r"\b(master|m\.?s\.?|m\.?sc|mba|m\.?tech)\b"),
                           ("Bachelors", r"\b(bachelor|b\.?s\.?|b\.?sc|b\.?tech|b\.?a\.?)\b"), ("Diploma", r"\b(diploma|associate)\b")]:
        if re.search(pattern, text_lower):
            return level
    return "Not detected"


def extract_experience(text):
    for pattern in [r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)", r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:in|of|working)"]:
        matches = re.findall(pattern, text.lower())
        years = [int(m) for m in matches if int(m) <= 50]
        if years:
            return max(years)
    return None


def compute_match_score(resume_skills, jd_skills):
    if not jd_skills:
        return 0, [], []
    r_set = set(s.lower() for s, _ in resume_skills)
    j_set = set(s.lower() for s, _ in jd_skills)
    matched = r_set.intersection(j_set)
    missing = j_set - r_set
    score = len(matched) / len(j_set) * 100 if j_set else 0
    return min(score, 100), list(matched), list(missing)


# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    resources = {}
    with open(os.path.join(DATA_PROCESSED, "label_encoder.pkl"), "rb") as f:
        resources["label_encoder"] = pickle.load(f)

    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if not os.path.exists(tfidf_path):
        tfidf_path = os.path.join(DATA_PROCESSED, "tfidf_vectorizer.pkl")
    with open(tfidf_path, "rb") as f:
        resources["tfidf"] = pickle.load(f)

    baseline_path = os.path.join(MODELS_DIR, "best_baseline.pkl")
    if os.path.exists(baseline_path):
        with open(baseline_path, "rb") as f:
            resources["classifier"] = pickle.load(f)

    with open(os.path.join(DATA_PROCESSED, "skills_taxonomy.json"), "r") as f:
        resources["skills_taxonomy"] = json.load(f)

    profiles_path = os.path.join(
        DATA_PROCESSED, "category_skill_profiles.json")
    if os.path.exists(profiles_path):
        with open(profiles_path, "r") as f:
            resources["category_profiles"] = json.load(f)
    else:
        resources["category_profiles"] = {}

    try:
        from sentence_transformers import SentenceTransformer
        resources["sbert"] = SentenceTransformer("all-MiniLM-L6-v2")
    except:
        resources["sbert"] = None

    return resources


# ============================================================
# STREAMLIT PAGES
# ============================================================
def main():
    st.sidebar.image(
        "https://img.icons8.com/3d-fluency/94/artificial-intelligence.png", width=80)
    st.sidebar.title("AI-Powered ATS")
    st.sidebar.markdown("**Master Thesis — V2**")
    st.sidebar.markdown("Hitik Sharma | 2026")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Supported:** PDF | DOCX | TXT")
    st.sidebar.markdown("**New:** 6-Layer Credibility Gate")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigation", [
                            "🏠 Home", "📄 Analyze Resume", "🔗 Match Resume to JD", "📊 Dashboard"])

    with st.spinner("Loading AI models..."):
        resources = load_models()

    if page == "🏠 Home":
        show_home()
    elif page == "📄 Analyze Resume":
        show_analyze(resources)
    elif page == "🔗 Match Resume to JD":
        show_match(resources)
    elif page == "📊 Dashboard":
        show_dashboard(resources)


def show_home():
    st.title("🎯 AI-Powered Applicant Tracking System")
    st.markdown(
        "### Intelligent Resume Screening with Deep Learning & Explainable AI")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📄 Analyze Resume")
        st.markdown("Upload a resume (PDF/DOCX/TXT). Get AI analysis with **6-layer credibility verification** — rejects non-resumes, flags cover letters.")
    with col2:
        st.markdown("#### 🔗 Match to Job")
        st.markdown(
            "Compare resume against job description. Get skill match, semantic similarity, and gap analysis.")
    with col3:
        st.markdown("#### 🛡️ Credibility Gate")
        st.markdown("Every document passes through **6 validation layers** before classification — document type, skills, confidence, structure, consistency.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔬 Technology Stack")
        st.markdown(
            "**Models:** Linear SVM (100%), BiLSTM+Attention (98.97%), Multi-Task Scorer (98.97%)")
        st.markdown("**NLP:** SBERT, TF-IDF, spaCy, 200+ Skill Taxonomy")
    with col2:
        st.markdown("#### 🆕 What's New in V2")
        st.markdown("✅ Cover letter detection and rejection")
        st.markdown("✅ Non-resume document filtering")
        st.markdown("✅ 6-layer credibility validation")
        st.markdown("✅ Category-skill consistency checking")


def show_analyze(resources):
    st.title("📄 Resume Analysis")
    st.markdown(
        "Upload a resume to get AI-powered analysis with credibility verification.")

    resume_text = show_resume_uploader(key_prefix="analyze")

    if st.button("🔍 Analyze Resume", type="primary") and resume_text.strip():
        with st.spinner("🧠 Running 6-layer credibility check + AI analysis..."):
            cleaned = clean_text(resume_text)

            # Run credibility check FIRST
            cred = credibility_check(cleaned, resources["tfidf"], resources["classifier"],
                                     resources["skills_taxonomy"], resources["label_encoder"])

            # Display credibility verdict prominently
            st.markdown("---")
            st.markdown("### 🛡️ Credibility Verification")

            # Verdict banner
            if cred['verdict'] == 'REJECTED':
                st.error(f"❌ **REJECTED** — {cred['verdict_msg']}")
                st.markdown(
                    f"**Document Type Detected:** {cred['document_type'].replace('_', ' ').title()}")
                st.markdown(
                    f"**Credibility Score:** {cred['credibility_score']}% ({cred['layers_passed']}/{cred['total_layers']} layers passed)")

                st.markdown("#### 🚩 Issues Found:")
                for severity, flag_name, detail in cred['flags']:
                    icon = "🔴" if severity == "HIGH" else "🟡"
                    st.warning(
                        f"{icon} **[{severity}] {flag_name}:** {detail}")

                st.info("💡 This document was not classified because it does not meet minimum resume requirements. Please upload a valid resume (PDF, DOCX, or TXT).")
                return  # STOP — don't show classification for rejected documents

            elif cred['verdict'] == 'MANUAL REVIEW':
                st.warning(
                    f"⚠️ **MANUAL REVIEW RECOMMENDED** — {cred['verdict_msg']}")
            elif cred['verdict'] == 'ACCEPT WITH REVIEW':
                st.info(
                    f"ℹ️ **ACCEPTED WITH ADVISORY** — {cred['verdict_msg']}")
            else:
                st.success(f"✅ **ACCEPTED** — {cred['verdict_msg']}")

            # Show layer results
            cols = st.columns(6)
            layer_names = ['Length', 'Doc Type', 'Skills',
                           'Confidence', 'Structure', 'Consistency']
            for i, (name, col) in enumerate(zip(layer_names, cols)):
                passed = i < cred['layers_passed'] or (i >= cred['layers_passed'] and
                                                       not any(f[1] in ['INSUFFICIENT_LENGTH', 'COVER_LETTER', 'NOT_A_RESUME',
                                                                        'FEW_SKILLS', 'LOW_CONFIDENCE', 'NO_STRUCTURE', 'CATEGORY_MISMATCH']
                                                               and layer_names.index(name) == ['INSUFFICIENT_LENGTH', 'COVER_LETTER', 'NOT_A_RESUME',
                                                                                               'FEW_SKILLS', 'LOW_CONFIDENCE', 'NO_STRUCTURE', 'CATEGORY_MISMATCH'].index(f[1])
                                                               for f in cred['flags']))
                # Simplified: check if this layer's flag exists
                layer_flag_map = {0: 'INSUFFICIENT_LENGTH', 1: ['COVER_LETTER', 'NOT_A_RESUME'],
                                  2: 'FEW_SKILLS', 3: 'LOW_CONFIDENCE', 4: 'NO_STRUCTURE', 5: 'CATEGORY_MISMATCH'}
                flag_names_in_result = [f[1] for f in cred['flags']]

                expected = layer_flag_map[i]
                if isinstance(expected, list):
                    layer_passed = not any(
                        e in flag_names_in_result for e in expected)
                else:
                    layer_passed = expected not in flag_names_in_result

                with col:
                    if layer_passed:
                        st.metric(name, "✅ Pass")
                    else:
                        st.metric(name, "❌ Fail")

            # Now show classification results (only if not rejected)
            st.markdown("---")
            st.markdown("### 📊 Classification Results")

            contact = extract_contact(resume_text)
            education = extract_education(resume_text)
            experience = extract_experience(resume_text)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🏷️ Category", cred['predicted_category'])
            c2.metric("📈 Confidence", f"{cred['confidence']:.0f}%")
            c3.metric("🛠️ Skills", cred['skills_count'])
            c4.metric("🛡️ Credibility", f"{cred['credibility_score']}%")

            st.markdown("---")

            # Two columns: skills + contact info
            left, right = st.columns([3, 2])

            with left:
                st.markdown("#### 🛠️ Skills Extracted")
                if cred['skills_found']:
                    skill_by_cat = {}
                    for skill, cat in cred['skills_found']:
                        skill_by_cat.setdefault(cat.replace(
                            "_", " ").title(), []).append(skill)
                    for cat, cat_skills in skill_by_cat.items():
                        st.markdown(f"**{cat}:** {', '.join(cat_skills)}")

                # Top 3 predictions
                st.markdown("#### 🎯 Top 3 Predictions")
                for cat, prob in cred['top3']:
                    st.progress(prob/100, text=f"{cat}: {prob}%")

                # Flags
                if cred['flags']:
                    st.markdown("#### 🚩 Flags")
                    for severity, flag_name, detail in cred['flags']:
                        icon = "🔴" if severity == "HIGH" else "🟡"
                        st.warning(f"{icon} {detail}")
                else:
                    st.success("✅ No issues detected!")

            with right:
                st.markdown("#### 📞 Contact Info")
                st.markdown(
                    f"**Email:** {contact.get('email') or '❌ Not found'}")
                st.markdown(
                    f"**Phone:** {contact.get('phone') or '❌ Not found'}")
                st.markdown(
                    f"**LinkedIn:** {contact.get('linkedin') or '❌ Not found'}")
                st.markdown(
                    f"**GitHub:** {contact.get('github') or '❌ Not found'}")

                st.markdown("#### 🎓 Education")
                st.info(f"**{education}**")

                st.markdown("#### 💼 Experience")
                st.info(
                    f"**{experience} years**" if experience else "Not explicitly mentioned")

                st.markdown("#### 📋 Document Type")
                st.info(
                    f"**{cred['document_type'].replace('_', ' ').title()}** ({cred['sections_detected']} sections detected)")


def show_match(resources):
    st.title("🔗 Resume-Job Description Matching")

    col1, col2 = st.columns(2)
    with col1:
        resume_text = show_resume_uploader(key_prefix="match")
    with col2:
        st.markdown("#### 💼 Job Description")
        jd_text = st.text_area("Paste Job Description:", height=300, key="match_jd",
                               placeholder="Paste the full job description here...")

    if st.button("🔗 Calculate Match", type="primary") and resume_text.strip() and jd_text.strip():
        with st.spinner("🧠 Computing match..."):
            resume_clean = clean_text(resume_text)
            jd_clean = clean_text(jd_text)

            # Credibility check on resume first
            cred = credibility_check(resume_clean, resources["tfidf"], resources["classifier"],
                                     resources["skills_taxonomy"], resources["label_encoder"])

            if cred['verdict'] == 'REJECTED':
                st.error(f"❌ Document rejected: {cred['verdict_msg']}")
                return

            # Extract skills from both
            resume_skills = cred['skills_found']

            jd_skills = []
            jd_lower = jd_clean.lower()
            for category, skills in resources["skills_taxonomy"].items():
                for skill in skills:
                    if len(skill) <= 3:
                        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', jd_lower):
                            jd_skills.append((skill, category))
                    elif skill.lower() in jd_lower:
                        jd_skills.append((skill, category))

            match_score, matched, missing = compute_match_score(
                resume_skills, jd_skills)

            # SBERT similarity
            sbert_score = 0
            if resources.get("sbert"):
                emb = resources["sbert"].encode(
                    [resume_clean[:2000], jd_clean[:2000]])
                sbert_score = float(np.dot(
                    emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))) * 100

            final_match = 0.5 * match_score + 0.5 * \
                sbert_score if sbert_score else match_score

            st.markdown("---")
            st.markdown("### 📊 Match Results")

            c1, c2, c3 = st.columns(3)
            c1.metric("🛠️ Skill Match", f"{match_score:.0f}%")
            c2.metric("🧠 Semantic Match", f"{sbert_score:.0f}%")
            delta = "Strong ✅" if final_match >= 70 else "Moderate 📋" if final_match >= 40 else "Weak ⚠️"
            c3.metric("🎯 Overall", f"{final_match:.0f}%", delta=delta)

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### ✅ Matched Skills")
                for s in matched[:15]:
                    st.markdown(f"✅ {s}")
                if not matched:
                    st.info("No matches")
            with col2:
                st.markdown("#### ❌ Missing Skills")
                for s in missing[:15]:
                    st.markdown(f"❌ {s}")
                if not missing:
                    st.success("All skills present!")
            with col3:
                st.markdown("#### 💡 Extra Skills")
                extra = set(s.lower() for s, _ in resume_skills) - \
                    set(s.lower() for s, _ in jd_skills)
                for s in list(extra)[:15]:
                    st.markdown(f"💡 {s}")

            st.markdown("---")
            if final_match >= 70:
                st.success(f"🎯 **STRONG MATCH** ({final_match:.0f}%)")
            elif final_match >= 40:
                st.warning(f"📋 **MODERATE MATCH** ({final_match:.0f}%)")
            else:
                st.error(f"⚠️ **WEAK MATCH** ({final_match:.0f}%)")

            if missing:
                st.info(
                    f"💡 **To improve:** Develop these skills: **{', '.join(list(missing)[:5])}**")


def show_dashboard(resources):
    st.title("📊 ATS Dashboard")

    st.markdown("### 🏆 Model Performance")
    perf = pd.DataFrame({
        "Model": ["Linear SVM (TF-IDF)", "Gradient Boosting (TF-IDF)", "BiLSTM + Attention", "Multi-Task Scorer", "SVM + SBERT"],
        "Accuracy": ["100.0%", "100.0%", "98.97%", "98.97%", "98.97%"],
        "F1-Score": ["1.0000", "1.0000", "0.9896", "0.9903", "0.9900"],
        "Type": ["Baseline", "Baseline", "Deep Learning", "Deep Learning", "Baseline"],
    })
    st.dataframe(perf, use_container_width=True)

    st.markdown("### 🛡️ Credibility Gate — 6 Layers")
    layers_df = pd.DataFrame({
        "Layer": ["1. Length", "2. Document Type", "3. Skills", "4. Confidence", "5. Structure", "6. Consistency"],
        "Check": ["≥40 words", "Resume vs Cover Letter vs Other", "≥3 professional skills", "≥50% confidence", "≥2 resume sections", "Skills match category"],
        "Rejects": ["Empty/minimal docs", "Bible, recipes, news, cover letters", "Generic text without skills", "Ambiguous predictions", "Unstructured documents", "Mismatched classifications"],
    })
    st.dataframe(layers_df, use_container_width=True)

    st.markdown("### 🛠️ Skills Taxonomy")
    taxonomy = resources["skills_taxonomy"]
    total = sum(len(v) for v in taxonomy.values())
    st.markdown(f"**{total} skills** across **{len(taxonomy)} categories**")
    cols = st.columns(3)
    for i, (cat, skills) in enumerate(taxonomy.items()):
        with cols[i % 3]:
            with st.expander(f"{cat.replace('_', ' ').title()} ({len(skills)})"):
                st.write(", ".join(skills))

    st.markdown("### 📋 Supported Categories (25)")
    cats = list(resources["label_encoder"].classes_)
    cols = st.columns(5)
    for i, cat in enumerate(cats):
        with cols[i % 5]:
            st.markdown(f"• {cat}")


if __name__ == "__main__":
    main()
