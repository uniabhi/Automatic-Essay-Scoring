from flask import Flask, request, render_template, redirect, url_for
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from werkzeug.utils import secure_filename
import os
import PyPDF2
import docx

from flask import Flask

app = Flask(__name__, template_folder='C:/Users/uniab/FREELANCING/LLM_AUTOMATIC GRADING/PAVAN-LLM-FINAL_This_Week/coding/')

# Configurations
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base")

# Function to provide feedback based on score
def get_feedback(score):
    if score > 0.8:
        return "Excellent work! Keep up the great writing."
    elif score > 0.6:
        return "Good job! However, there's room for improvement in clarity and structure."
    elif score > 0.4:
        return "Fair attempt, but consider improving your argumentation and grammar."
    else:
        return "Needs significant improvement. Focus on organizing your thoughts and improving sentence structure."

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    text = ""
    
    if ext == 'pdf':
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfFileReader(f)
            for page in range(reader.numPages):
                text += reader.getPage(page).extract_text()
    
    elif ext == 'docx':
        doc = docx.Document(filepath)
        for para in doc.paragraphs:
            text += para.text
    
    elif ext == 'txt':
        with open(filepath, 'r') as f:
            text = f.read()
    
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    feedback = ""
    score = None
    essay_text = ""

    if request.method == 'POST':
        if 'essay_file' in request.files:
            file = request.files['essay_file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                essay_text = extract_text_from_file(filepath)
                os.remove(filepath)  # Clean up uploaded file after processing
            else:
                feedback = "Invalid file format. Please upload a .pdf, .docx, or .txt file."
        
        if not essay_text:
            essay_text = request.form['essay']
        
        if len(essay_text) < 50:
            feedback = "Essay text is too short. Please write at least 50 characters."
        else:
            inputs = tokenizer(essay_text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                scores = outputs.logits.squeeze().detach().numpy()
                score = scores.mean()  # Averaging across all outputs; adjust as needed
                feedback = get_feedback(score)
        
    return render_template('index.html', score=score, essay_text=essay_text, feedback=feedback)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)