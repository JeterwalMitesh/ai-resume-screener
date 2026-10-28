import streamlit as st
import pdfplumber
import pypdfium2 as pdfium
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="AI Resume Screener & Skill-Gap Analyzer", layout="wide")

st.title("AI Resume Screener & Skill-Gap Analyzer")
st.write("Apna resume upload karein aur Job Description ke according matching score aur recommended roadmaps dekhein.")

col_input1, col_input2 = st.columns(2)

with col_input1:
    job_description = st.text_area(
        "Job Description yahan paste karein:",
        placeholder="Company ki complete Job Description paste karein...",
        height=200
    )

with col_input2:
    uploaded_file = st.file_uploader("Apna Resume upload karein (PDF)", type=["pdf"])

# Dual-Engine Extractor: Digital Text + Fallback OCR
def extract_pdf_text(file):
    text = ""
    # Method 1: Direct digital text extraction
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
    except Exception:
        pass
        
    text = text.strip()
    
    # Method 2: Fallback to OCR if direct extraction fails (for scanned/photo PDFs)
    if not text:
        try:
            file.seek(0)
            pdf = pdfium.PdfDocument(file.read())
            ocr_text = ""
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                image = page.render(scale=2).to_pil()
                ocr_text += pytesseract.image_to_string(image) + " "
            text = ocr_text.strip()
        except Exception:
            text = ""
            
    return text

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()

def match_skill(skill, text):
    pattern = r'(?<!\w)' + re.escape(skill) + r'(?!\w)'
    return bool(re.search(pattern, text))

COMPREHENSIVE_SKILLS = [
    # Data Science & Analytics
    "python", "sql", "mysql", "postgresql", "mongodb", "pandas", "numpy", 
    "scikit learn", "tensorflow", "pytorch", "keras", "machine learning", 
    "deep learning", "nlp", "computer vision", "tableau", "power bi", "excel", 
    "statistics", "data visualization", "big data", "spark", "hadoop",
    
    # Web & Software Development
    "java", "c++", "c#", "javascript", "typescript", "html", "css", 
    "react", "angular", "vue", "node js", "django", "flask", "fastapi", 
    "spring boot", "rest api", "graphql",
    
    # Cloud, DevOps & Tools
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github", 
    "ci cd", "linux", "jira"
]

if st.button("Analyze Resume", type="primary"):
    if not job_description.strip():
        st.warning("Kripya pehle Job Description paste karein!")
    elif uploaded_file is None:
        st.warning("Kripya apna Resume PDF upload karein!")
    else:
        with st.spinner("Resume process ho raha hai (Scanned hone par OCR scan chalega)..."):
            resume_text = extract_pdf_text(uploaded_file)
        
        if not resume_text:
            st.error("Is PDF se text identify nahi ho saka. Kripya file check karein.")
        else:
            # 1. Text Normalization
            resume_clean = clean_text(resume_text)
            jd_clean = clean_text(job_description)
            
            # Word boundary matching
            required_skills = [s for s in COMPREHENSIVE_SKILLS if match_skill(s, jd_clean)]
            matched_skills = [s for s in required_skills if match_skill(s, resume_clean)]
            missing_skills = [s for s in required_skills if not match_skill(s, resume_clean)]
            
            # 2. Context Similarity via TF-IDF
            documents = [resume_clean, jd_clean]
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(documents)
            raw_cosine = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # 3. Hybrid Score Calculation
            if required_skills:
                skill_score = len(matched_skills) / len(required_skills)
                final_score = (skill_score * 0.75) + (raw_cosine * 0.25)
            else:
                final_score = raw_cosine
                
            final_percentage = round(final_score * 100, 2)
            
            # Overview UI
            st.markdown("---")
            st.subheader("Analysis Overview")
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            col_metric1.metric("Match Score", f"{final_percentage}%")
            col_metric2.metric("Matched Skills Count", len(matched_skills))
            col_metric3.metric("Missing Skills Count", len(missing_skills))
            
            if final_percentage >= 60:
                st.success("Verdict: Strong Profile Match for this opening.")
            elif final_percentage >= 35:
                st.warning("Verdict: Moderate Match. Kuch key skills cover karke score badhaya ja sakta hai.")
            else:
                st.error("Verdict: Low Match. Required profile ke sath skill mismatch hai.")
                
            st.markdown("---")
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.markdown("### Matched Skills")
                if matched_skills:
                    for s in matched_skills:
                        st.write(f"• **{s.title()}**")
                else:
                    st.write("Job Description ki koi listed technical skill match nahi hui.")
                    
            with col_res2:
                st.markdown("### Skill-Gap & Learning Recommendations")
                if missing_skills:
                    for s in missing_skills:
                        search_url = f"https://www.youtube.com/results?search_query={s.replace(' ', '+')}+full+course"
                        st.markdown(f"• **{s.title()}** — [Free Course Dekhein]({search_url})")
                else:
                    st.write("Badhiya! Job Description ki sabhi identified skills resume mein hain.")
            
            # Debug tool: Extracted text preview
            with st.expander("Resume se extract hua text dekhein (Debug Preview)"):
                st.write(resume_text)
