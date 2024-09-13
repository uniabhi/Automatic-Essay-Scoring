# Automatic Essay Grading System using DeBERTa v3 Base LLM

This project implements an automatic essay grading system using the DeBERTa v3 Base Large Language Model (LLM). The system is designed to assist educators by automating the process of grading student essays, providing consistent and efficient evaluations.

## Features

- Utilizes the DeBERTa v3 Base LLM for essay scoring
- Implements text preprocessing techniques for improved accuracy
- Generates embeddings using the all-MiniLM-L6-v2 model
- Fine-tunes the LLM on a specific essay dataset
- Provides a web-based user interface for easy essay submission and grading
- Supports multiple input formats: direct text input, PDF, Word, and TXT files
- Offers feedback along with the essay score

## Technologies Used

- Python 3.8
- PyTorch
- Transformers library (Hugging Face)
- Flask for API and web application
- HTML/CSS for frontend

## Setup and Installation

1. Clone the repository:
   ```
   https://github.com/uniabhi/Automatic-Essay-Scoring.git
   ```

2. Create a virtual environment and activate it:
   ```
   conda create -n essay_grading python=3.8
   conda activate essay_grading
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the fine-tuned model weights and tokenizer files (not included in the repository due to size constraints) and place them in the appropriate directories:
   - Place model weights in the `results` folder
   - Place tokenizer files in the `fine_tuned_model` folder

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://127.0.0.1:5000/`

3. Use the web interface to submit essays for grading:
   - Paste the essay text directly into the text box
   - Upload a PDF, Word, or TXT file containing the essay

4. The system will process the essay and return a grade along with feedback

## Project Structure

- `app.py`: Main Flask application
- `index.html`: Frontend HTML template
- `style.css`: CSS styles for the frontend
- `results/`: Directory containing fine-tuned model weights
- `fine_tuned_model/`: Directory containing tokenizer files
- `requirements.txt`: List of Python dependencies

5. User Interface
![image](https://github.com/user-attachments/assets/8c3e1e2a-d924-42e3-884d-ee9082ba88a3)

This image showcases the user interface of our Automated Essay Grading system, demonstrating the text input area, grading functionality, and score display.

## Future Work

- Implement more advanced feedback mechanisms
- Integrate with Learning Management Systems (LMS)
- Improve model performance through advanced fine-tuning techniques
- Conduct longitudinal studies on the impact of automated grading

## Contributors

- ABHISHEK KUMAR

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The Hewlett Foundation for providing the initial dataset
- Hugging Face for the Transformers library
- The open-source community for various tools and libraries used in this project
