from flask import Flask, request, jsonify, render_template
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import re
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os
import sqlite3
from datetime import datetime
import ssl
import shutil
from urllib.request import urlretrieve
import zipfile
import socket

# ===== COMPLETE NLTK SOLUTION =====
def setup_nltk():
    nltk_data_path = r'D:\resume_nltk_data'
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path = [nltk_data_path]
    
    # Bypass SSL and network issues
    ssl._create_default_https_context = ssl._create_unverified_context
    socket.setdefaulttimeout(30)
    
    def install_resource(resource_name, package_name):
        try:
            # Try official download first
            nltk.download(package_name, download_dir=nltk_data_path)
            
            # Special handling for punkt
            if package_name == 'punkt':
                src = os.path.join(nltk_data_path, 'tokenizers', 'punkt')
                dst = os.path.join(nltk_data_path, 'tokenizers', 'punkt_tab')
                if not os.path.exists(dst) and os.path.exists(src):
                    shutil.copytree(src, dst)
            return True
        except:
            # Fallback to direct GitHub download
            try:
                url = f"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/{resource_name.split('/')[0]}/{package_name}.zip"
                target_dir = os.path.join(nltk_data_path, *resource_name.split('/'))
                os.makedirs(target_dir, exist_ok=True)
                
                urlretrieve(url, "temp.zip")
                with zipfile.ZipFile("temp.zip", 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                os.remove("temp.zip")
                
                # Create punkt_tab alias if needed
                if package_name == 'punkt':
                    src = os.path.join(nltk_data_path, 'tokenizers', 'punkt')
                    dst = os.path.join(nltk_data_path, 'tokenizers', 'punkt_tab')
                    if not os.path.exists(dst) and os.path.exists(src):
                        shutil.copytree(src, dst)
                return True
            except Exception as e:
                print(f"Failed to install {package_name}: {str(e)}")
                return False

    # Install required resources
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords')
    ]
    
    for resource, package in resources:
        if not install_resource(resource, package):
            print(f"FATAL: Could not install {package}")
            exit(1)

    # Final verification
    required_paths = [
        os.path.join(nltk_data_path, 'tokenizers', 'punkt'),
        os.path.join(nltk_data_path, 'tokenizers', 'punkt_tab'),
        os.path.join(nltk_data_path, 'corpora', 'stopwords')
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            print(f"FATAL: Resource missing at {path}")
            exit(1)

# Run setup before creating Flask app
setup_nltk()
# ===== END NLTK SOLUTION =====

app = Flask(__name__)

# Initialize NLP
nlp = spacy.load("en_core_web_lg")

# Database setup
def init_db():
    conn = sqlite3.connect('resume_matcher.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analyses
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  resume_text TEXT,
                  job_desc TEXT,
                  similarity_score REAL,
                  missing_skills TEXT,
                  timestamp DATETIME)''')
    conn.commit()
    conn.close()

init_db()

# Text preprocessing (resources are now guaranteed to exist)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in tokens if word not in stop_words)

# Extract skills from text
def extract_skills(text):
    doc = nlp(text)
    skills = []
    
    # Pattern matching for skills
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3:  # Skills are usually 1-3 words
            skills.append(chunk.text)
    
    # Additional filtering
    skills = list(set([skill for skill in skills if skill not in stopwords.words('english')]))
    return skills

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Calculate similarity
def calculate_similarity(resume_text, job_desc_text):
    # TF-IDF approach
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_desc_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    
    return round(similarity * 100, 2)

# Find missing skills
def find_missing_skills(resume_skills, job_desc_skills):
    return list(set(job_desc_skills) - set(resume_skills))

# Generate suggestions
def generate_suggestions(missing_skills):
    suggestions = []
    skill_resources = {
        "python": ["Codecademy Python Course", "Python for Everybody on Coursera"],
        "machine learning": ["Andrew Ng's ML Course", "Fast.ai Practical Deep Learning"],
        "sql": ["SQLZoo interactive tutorials", "Mode Analytics SQL Tutorial"],
    }
    
    for skill in missing_skills:
        if skill.lower() in skill_resources:
            suggestions.append({
                "skill": skill,
                "resources": skill_resources[skill.lower()]
            })
        else:
            suggestions.append({
                "skill": skill,
                "resources": ["General online courses like Udemy, Coursera for " + skill]
            })
    
    return suggestions

# API Endpoints
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files or 'job_desc' not in request.form:
        return jsonify({"error": "Missing resume or job description"}), 400
    
    try:
        # Process files
        resume_file = request.files['resume']
        job_desc_text = request.form['job_desc']
        
        # Extract text
        resume_text = extract_text_from_pdf(resume_file)
        preprocessed_resume = preprocess_text(resume_text)
        preprocessed_job_desc = preprocess_text(job_desc_text)
        
        # Extract skills
        resume_skills = extract_skills(preprocessed_resume)
        job_desc_skills = extract_skills(preprocessed_job_desc)
        
        # Calculate metrics
        similarity_score = calculate_similarity(preprocessed_resume, preprocessed_job_desc)
        missing_skills = find_missing_skills(resume_skills, job_desc_skills)
        suggestions = generate_suggestions(missing_skills)
        
        # Store analysis in DB
        conn = sqlite3.connect('resume_matcher.db')
        c = conn.cursor()
        c.execute("INSERT INTO analyses (resume_text, job_desc, similarity_score, missing_skills, timestamp) VALUES (?, ?, ?, ?, ?)",
                 (resume_text[:1000], job_desc_text[:1000], similarity_score, str(missing_skills), datetime.now()))
        conn.commit()
        conn.close()
        
        # Prepare response
        response = {
            "similarity_score": similarity_score,
            "resume_skills": resume_skills,
            "job_desc_skills": job_desc_skills,
            "missing_skills": missing_skills,
            "suggestions": suggestions,
            "status": "success"
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route('/history')
def get_history():
    conn = sqlite3.connect('resume_matcher.db')
    c = conn.cursor()
    c.execute("SELECT id, similarity_score, timestamp FROM analyses ORDER BY timestamp DESC LIMIT 10")
    history = c.fetchall()
    conn.close()
    
    return jsonify([{"id": row[0], "score": row[1], "timestamp": row[2]} for row in history])

if __name__ == '__main__':
    app.run(debug=True)
