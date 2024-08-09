from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from werkzeug.utils import secure_filename
import os
import PyPDF2
import docx

app = Flask(__name__, template_folder='C:/Users/uniab/FREELANCING/LLM_AUTOMATIC GRADING/PAVAN-LLM-FINAL_This_Week/coding/')

# Configurations
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_dir = "results"  # Directory containing the model
tokenizer_dir = "fine_tuned_model"  # Directory containing the tokenizer

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Function to provide feedback based on score
def get_feedback(score):
    if score > 5.0:
        return "Feedback: Excellent work! Keep up the great writing."
    elif score > 3.0:
        return "Feedback: Good job! However, there's room for improvement in clarity and structure."
    elif score > 1.0:
        return "Feedback: Fair attempt, but consider improving your argumentation, more essay words and grammar."
    else:
        return "Feedback: Needs significant improvement. Focus on organizing your thoughts, more essay words and improving sentence structure."

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    text = ""
    
    if ext == 'pdf':
        try:
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in range(len(reader.pages)):
                    text += reader.pages[page].extract_text()
        except Exception as e:
            print(f"Error reading PDF file: {e}")
    
    elif ext == 'docx':
        try:
            doc = docx.Document(filepath)
            for para in doc.paragraphs:
                text += para.text
        except Exception as e:
            print(f"Error reading DOCX file: {e}")
    
    elif ext == 'txt':
        try:
            with open(filepath, 'r') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading TXT file: {e}")
    
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
            essay_text = request.form.get('essay', '')

        if len(essay_text) < 50:
            feedback = "Essay text is too short. Please write at least 50 characters."
        else:
            inputs = tokenizer(essay_text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                scores = outputs.logits.squeeze().detach().numpy()
                score = round(scores.mean(), 2)  # Averaging across all outputs and rounding off to two decimal places
                feedback = get_feedback(score)
        
    return render_template('index.html', score=score, essay_text=essay_text, feedback=feedback)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
