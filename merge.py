import streamlit as st
import PyPDF2
import pytesseract as tess
from PIL import Image
import google.generativeai as genai
import io
import os
import json
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import time
import re
from datetime import timedelta

# Configure Google Gemini
genai.configure(api_key="YOUR_GOOGLE_API_KEY")  # Replace with your actual API key
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 2048,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel('gemini-pro', generation_config=generation_config, safety_settings=safety_settings)

class AdvancedQuizGenerator:
    def __init__(self):
        self.default_settings = {
            "num_questions": 10,
            "num_options": 4,
            "difficulty_levels": {"Easy": 30, "Medium": 50, "Hard": 20},  # Percentage distribution
            "time_limit": 60,
            "passing_score": 75,
            "total_marks": 100,
            "question_types": ["MCQ", "True/False", "Fill-in-the-blank"]
        }
        
        if "settings" not in st.session_state:
            st.session_state.settings = self.default_settings.copy()
        if "questions" not in st.session_state:
            st.session_state.questions = []
        if "quiz_active" not in st.session_state:
            st.session_state.quiz_active = False

    def extract_text(self, uploaded_file):
        """Extract text from various file types including images using OCR"""
        text = ""
        
        if uploaded_file.type.startswith('image'):
            try:
                image = Image.open(io.BytesIO(uploaded_file.getvalue()))
                img_array = np.array(image)
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                text = tess.image_to_string(gray)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                return None
        elif uploaded_file.type == "application/pdf":
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
                text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            except Exception as e:
                st.error(f"Error reading PDF: {str(e)}")
                return None
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
                text = "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                st.error(f"Error reading DOCX: {str(e)}")
                return None
        return text

    def generate_question_with_gemini(self, context):
        """Generate questions using Google Gemini with advanced prompt engineering"""
        prompt = f"""
        Generate a comprehensive exam-level question based on the following context:
        {context}
        
        Requirements:
        - Question must test higher-order thinking skills
        - Include 1 correct answer and {st.session_state.settings['num_options']-1} plausible distractors
        - Options should be mutually exclusive
        - Mark correct answer with [CORRECT]
        - Difficulty level: {self.get_difficulty_label()}
        - Question type: {np.random.choice(st.session_state.settings['question_types'], p=[0.7, 0.2, 0.1])}
        
        Format:
        Question: [Your question here]
        Options:
        A) [Option 1] [CORRECT]
        B) [Option 2]
        C) [Option 3]
        D) [Option 4]
        """
        
        try:
            response = model.generate_content(prompt)
            return self.parse_gemini_response(response.text)
        except Exception as e:
            st.error(f"Error generating question: {str(e)}")
            return None

    def parse_gemini_response(self, response_text):
        """Parse Gemini response into structured question format"""
        question_pattern = re.compile(r"Question:\s*(.+?)\nOptions:", re.DOTALL)
        options_pattern = re.compile(r"[A-Z]\)\s*(.+?)(\s*\[CORRECT\])?", re.MULTILINE)
        
        question_match = question_pattern.search(response_text)
        if not question_match:
            return None
            
        question = {
            "question": question_match.group(1).strip(),
            "options": [],
            "correct_index": -1
        }
        
        for idx, match in enumerate(options_pattern.finditer(response_text)):
            option_text = match.group(1).strip()
            is_correct = bool(match.group(2))
            question["options"].append(option_text)
            if is_correct:
                question["correct_index"] = idx
                
        if len(question["options"]) != st.session_state.settings['num_options'] or question["correct_index"] == -1:
            return None
            
        return question

    def generate_quiz(self, context):
        """Generate full quiz using advanced NLP techniques"""
        try:
            # Preprocess context using NLP techniques
            key_sentences = self.extract_key_concepts(context)
            
            # Generate questions using multiple techniques
            questions = []
            for sentence in key_sentences[:st.session_state.settings['num_questions']]:
                gemini_question = self.generate_question_with_gemini(sentence)
                if gemini_question:
                    questions.append(gemini_question)
                    
            return questions
        except Exception as e:
            st.error(f"Quiz generation failed: {str(e)}")
            return []

    def extract_key_concepts(self, text):
        """Extract key concepts using NLP and ML techniques"""
        # Using Gemini for concept extraction
        prompt = f"""
        Extract the most important conceptual sentences from this text for creating exam questions:
        {text}
        
        Return as JSON array:
        {{"sentences": ["sentence1", "sentence2", ...]}}
        """
        
        try:
            response = model.generate_content(prompt)
            data = json.loads(response.text)
            return data.get("sentences", [])
        except:
            # Fallback to TF-IDF based extraction
            return self.tfidf_extraction(text)

    def tfidf_extraction(self, text):
        """Fallback key concept extraction using TF-IDF"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        sentences = re.split(r'(?<=[.!?])\s+', text)
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(sentences)
        scores = np.array(X.sum(axis=1)).flatten()
        top_indices = scores.argsort()[-st.session_state.settings['num_questions']:][::-1]
        return [sentences[i] for i in top_indices]

    def run_quiz_interface(self):
        """Advanced quiz interface with real-time processing"""
        st.title("AI-Powered Adaptive Quiz System")
        
        # Settings sidebar
        with st.sidebar:
            st.header("Quiz Configuration")
            st.session_state.settings['num_questions'] = st.number_input("Number of Questions", 5, 50, 10)
            st.session_state.settings['difficulty_levels'] = {
                "Easy": st.slider("Easy %", 0, 100, 30),
                "Medium": st.slider("Medium %", 0, 100, 50),
                "Hard": st.slider("Hard %", 0, 100, 20)
            }
        
        # Main interface
        input_method = st.radio("Input Source:", ["Upload Document", "Enter Topic"])
        
        if input_method == "Upload Document":
            uploaded_file = st.file_uploader("Upload PDF/Image/DOCX", type=["pdf", "png", "jpg", "jpeg", "docx"])
            if uploaded_file:
                context = self.extract_text(uploaded_file)
        else:
            context = st.text_area("Enter Topic/Context:", height=200)
        
        if st.button("Generate Quiz") and context:
            with st.spinner("Generating advanced quiz using AI..."):
                st.session_state.questions = self.generate_quiz(context)
                st.session_state.quiz_active = True
                st.session_state.start_time = datetime.now()
                
        if st.session_state.quiz_active:
            self.display_quiz()
            
    def display_quiz(self):
        """Interactive quiz display with real-time validation"""
        st.header("Generated Quiz")
        time_remaining = timedelta(minutes=st.session_state.settings['time_limit']) - (datetime.now() - st.session_state.start_time)
        
        # Timer display
        st.write(f"Time Remaining: {time_remaining.seconds//60:02d}:{time_remaining.seconds%60:02d}")
        
        # Display questions
        answers = {}
        for i, q in enumerate(st.session_state.questions):
            with st.expander(f"Question {i+1}", expanded=True):
                st.markdown(f"**{q['question']}**")
                answer = st.radio("Options:", q['options'], key=f"q_{i}", index=None)
                answers[f"q_{i}"] = answer
                
        if st.button("Submit Quiz"):
            self.evaluate_quiz(answers)
            
    def evaluate_quiz(self, answers):
        """Advanced evaluation with analytics"""
        score = 0
        results = []
        
        for i, q in enumerate(st.session_state.questions):
            user_answer = answers.get(f"q_{i}")
            is_correct = (user_answer == q['options'][q['correct_index']])
            score += is_correct * (st.session_state.settings['total_marks']/len(st.session_state.questions))
            results.append({
                "Question": q['question'],
                "Your Answer": user_answer,
                "Correct Answer": q['options'][q['correct_index']],
                "Status": "Correct" if is_correct else "Incorrect"
            })
        
        # Display results
        st.subheader(f"Final Score: {score:.1f}/{st.session_state.settings['total_marks']}")
        st.write(pd.DataFrame(results))
        
        # Generate analytics
        self.generate_analytics(results)
        
    def generate_analytics(self, results):
        """Generate advanced learning analytics"""
        st.subheader("Performance Analytics")
        
        # Difficulty analysis
        difficulty_counts = {"Easy":0, "Medium":0, "Hard":0}
        for q in st.session_state.questions:
            difficulty_counts[self.get_difficulty_label()] += 1
        
        # Accuracy by difficulty
        st.write("### Accuracy by Difficulty Level")
        # Add visualization logic here
        
        # Knowledge gap analysis
        st.write("### Knowledge Gaps")
        incorrect_questions = [r for r in results if r['Status'] == "Incorrect"]
        if incorrect_questions:
            for q in incorrect_questions:
                st.write(f"**Concept to review:** {q['Question']}")
                st.write(f"Correct answer: {q['Correct Answer']}\n")
        else:
            st.success("Perfect score! No knowledge gaps detected.")

    def get_difficulty_label(self):
        """Get difficulty level based on configured distribution"""
        levels = list(st.session_state.settings['difficulty_levels'].keys())
        probs = [v/100 for v in st.session_state.settings['difficulty_levels'].values()]
        return np.random.choice(levels, p=probs)

if __name__ == "__main__":
    quiz = AdvancedQuizGenerator()
    quiz.run_quiz_interface()