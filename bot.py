import streamlit as st
import time
import json
import PyPDF2
import docx
import io
import random
import re
from datetime import datetime, timedelta
from transformers import pipeline

# Configure the page for better layout and JavaScript support
st.set_page_config(
    page_title="Enhanced Quiz Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EnhancedQuizGenerator:
    def __init__(self):
        # Initialize the question generation pipeline using a Hugging Face model.
        # Catch error if no deep learning backend is installed.
        try:
            self.qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")
        except RuntimeError as e:
            st.error(
                "Deep learning framework not found. Please install either PyTorch or TensorFlow.\n\n"
                "For PyTorch, run: `pip install torch torchvision torchaudio`\n"
                "For TensorFlow, see: https://www.tensorflow.org/install/"
            )
            raise e

        self.default_settings = {
            "num_questions": 5,
            "num_options": 4,
            "extra_questions": 0,
            "difficulty": "medium",
            "time_limit": 60,  # in minutes
            "passing_score": 70,  # percentage
            "total_marks": 100,
            "question_categories": ["U", "R", "A"]  # Understanding, Reasoning, Application
        }
        
        # Initialize session state
        if "settings" not in st.session_state:
            st.session_state.settings = self.default_settings.copy()
        if "questions" not in st.session_state:
            st.session_state.questions = []
        if "answers" not in st.session_state:
            st.session_state.answers = {}
        if "start_time" not in st.session_state:
            st.session_state.start_time = None
        if "quiz_active" not in st.session_state:
            st.session_state.quiz_active = False
            
    def extract_text_from_file(self, uploaded_file):
        """Extract text from uploaded PDF or DOCX file"""
        text = ""
        if uploaded_file.type == "application/pdf":
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            except Exception as e:
                st.error(f"Error reading PDF: {str(e)}")
                return None
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
                for para in doc.paragraphs:
                    text += para.text + "\n"
            except Exception as e:
                st.error(f"Error reading DOCX: {str(e)}")
                return None
        return text
            
    def render_settings(self):
        """Render enhanced quiz settings form"""
        st.sidebar.title("Quiz Settings")
        
        settings = st.session_state.settings
        
        # Basic settings
        settings["num_questions"] = st.sidebar.number_input(
            "Number of Main Questions",
            min_value=1,
            max_value=50,
            value=settings["num_questions"]
        )
        
        # Replace slider with radio buttons
        settings["num_options"] = st.sidebar.radio(
            "Options per Question",
            options=[2, 3, 4, 5, 6],
            index=[2, 3, 4, 5, 6].index(min(max(settings["num_options"], 2), 6))
        )
        
        settings["extra_questions"] = st.sidebar.number_input(
            "Number of Extra Questions",
            min_value=0,
            max_value=10,
            value=settings["extra_questions"]
        )
        
        # Marks and timing
        settings["total_marks"] = st.sidebar.number_input(
            "Total Marks",
            min_value=10,
            max_value=1000,
            value=settings["total_marks"]
        )
        
        settings["time_limit"] = st.sidebar.number_input(
            "Time Limit (minutes)",
            min_value=1,
            max_value=180,
            value=settings["time_limit"]
        )
        
        # Add passing score setting
        settings["passing_score"] = st.sidebar.slider(
            "Passing Score (%)",
            min_value=0,
            max_value=100,
            value=settings["passing_score"]
        )
        
        # Categories and difficulty
        settings["difficulty"] = st.sidebar.selectbox(
            "Overall Difficulty",
            options=["easy", "medium", "hard"],
            index=["easy", "medium", "hard"].index(settings["difficulty"])
        )
        
        if st.sidebar.button("Reset to Default"):
            st.session_state.settings = self.default_settings.copy()
            st.rerun()
    
    def extract_sentences(self, text, min_length=40):
        """Extract meaningful sentences from text content"""
        # Basic sentence splitting with regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out short sentences and clean them
        good_sentences = [s.strip() for s in sentences if len(s.strip()) >= min_length]
        return good_sentences
    
    def generate_question_from_text(self, text):
        # Ensure the text has enough context for the model
        if not text or len(text.split()) < 5:
            return "Not enough context to generate a question."
        try:
            # Create an input prompt for the model
            input_prompt = ("Generate a formal, grammatically correct exam-level multiple-choice question "
                            "with four distinct answer options based on the following text: " + text)
            result = self.qg_pipeline(input_prompt, max_length=64, do_sample=False)
            return result[0]['generated_text']
        except Exception as e:
            return f"Error generating question: {str(e)}"

    def generate_distractors(self, question_text):
        try:
            prompt = "Generate three plausible distractor options for the following question in a formal tone: " + question_text
            result = self.qg_pipeline(prompt, max_length=64, do_sample=False)
            # Assume the model returns a comma-separated list; split and trim the results.
            return [opt.strip() for opt in result[0]['generated_text'].split(',') if opt.strip()]
        except Exception as e:
            raise e

    def generate_quiz(self, content_source, is_file=False):
        """Generate quiz based on content source and current settings"""
        settings = st.session_state.settings
        total_questions = settings["num_questions"] + settings["extra_questions"]
        marks_per_question = settings["total_marks"] / settings["num_questions"]
        random.seed(time.time())
        sentences = []
        
        # Extract sentences from file or use topic as input
        if is_file and isinstance(content_source, str):
            sentences = self.extract_sentences(content_source)
        if sentences:
            random.shuffle(sentences)

        questions = []
        for i in range(total_questions):
            category = self.default_settings["question_categories"][i % 3]
            is_extra = i >= settings["num_questions"]

            # Generate question based on content
            if is_file and sentences and i < len(sentences):
                base_text = sentences[i]
                if len(base_text) > 100:
                    base_text = base_text[:97] + "..."
                question_data = self.generate_question_from_text(base_text)
            else:
                if isinstance(content_source, str) and not is_file:
                    question_data = self.generate_question_from_text(content_source)
                else:
                    question_data = f"Question #{i+1} from the uploaded document"

            # Parse the generated question and extract options
            try:
                # Example format: "Question? [Correct Answer] [Distractor1] [Distractor2] [Distractor3]"
                question_parts = re.split(r'\[|\]', question_data)
                question_text = question_parts[0].strip()
                correct_answer = question_parts[1].strip()
                distractors = [part.strip() for part in question_parts[2:] if part.strip()]
                
                # If distractors are insufficient, generate more using AI
                if len(distractors) < settings["num_options"] - 1:
                    additional_distractors = self.generate_distractors(question_text)
                    distractors.extend(additional_distractors)
                    distractors = distractors[:settings["num_options"] - 1]

                # Randomly insert the correct answer into the options
                options = distractors.copy()
                correct_index = random.randint(0, settings["num_options"] - 1)
                options.insert(correct_index, correct_answer)

            except Exception as e:
                # Fallback in case of errors
                options = [f"Option {j+1}" for j in range(settings["num_options"])]
                correct_index = 0

            # Create the question dictionary
            question = {
                "id": i,
                "question": question_text,
                "options": options,
                "correct_index": correct_index,
                "category": category,
                "marks": marks_per_question if not is_extra else 0,
                "is_extra": is_extra,
                "difficulty": settings["difficulty"]
            }
            questions.append(question)

        return questions
    
    def check_time_limit(self):
        """Check if time limit has been reached"""
        if st.session_state.start_time:
            elapsed = datetime.now() - st.session_state.start_time
            time_limit = timedelta(minutes=st.session_state.settings["time_limit"])
            if elapsed >= time_limit:
                st.session_state.quiz_active = False
                return True
        return False
    
    def display_timer(self):
        """Display real-time countdown timer using JavaScript"""
        if st.session_state.start_time:
            elapsed = datetime.now() - st.session_state.start_time
            time_limit = timedelta(minutes=st.session_state.settings["time_limit"])
            remaining = time_limit - elapsed

            if remaining.total_seconds() <= 0:
                st.error("⏰ Time's up!")
                return True

            minutes = int(remaining.total_seconds() // 60)
            seconds = int(remaining.total_seconds() % 60)

            # Use JavaScript for real-time countdown
            col1, col2 = st.columns([3, 1])
            with col2:
                timer_color = "red" if minutes < 5 else "orange" if minutes < 10 else "green"
                st.markdown(
                    f"""
                    <div id="timer" style="color: {timer_color}; font-size: 24px;">⏱️ {minutes:02d}:{seconds:02d}</div>
                    """,
                    unsafe_allow_html=True
                )

            # Inject JavaScript for real-time countdown
            st.markdown(
                """
                <script>
                function updateTimer() {
                    const timerElement = document.getElementById('timer');
                    if (!timerElement) return;

                    let timeString = timerElement.innerText;
                    let [minutes, seconds] = timeString.split(' ')[1].split(':').map(Number);

                    let totalSeconds = minutes * 60 + seconds - 1;
                    if (totalSeconds < 0) {
                        clearInterval(interval);
                        timerElement.style.color = 'red';
                        timerElement.innerText = "⏰ Time's up!";
                        alert('Time is up!');
                        window.location.reload();  // Reload the page to trigger result display
                        return;
                    }

                    minutes = Math.floor(totalSeconds / 60);
                    seconds = totalSeconds % 60;

                    timerElement.innerText = `⏱️ ${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
                }

                let interval = setInterval(updateTimer, 1000);
                </script>
                """,
                unsafe_allow_html=True
            )

            if remaining.total_seconds() <= 0:
                return True
            return False
        return False
    
    def run_quiz(self):
        """Main quiz interface with enhanced features"""
        st.title("Enhanced Quiz Generator")
        # Render settings sidebar
        self.render_settings()

        if not st.session_state.quiz_active:
            # Input method selection
            input_method = st.radio(
                "Select input method:",
                ["Enter Topic", "Upload Document"]
            )
            if input_method == "Enter Topic":
                topic = st.text_input("Enter quiz topic:")
                if st.button("Start Quiz") and topic:
                    st.session_state.questions = self.generate_quiz(topic)
                    st.session_state.start_time = datetime.now()
                    st.session_state.quiz_active = True
                    st.rerun()
            else:  # Upload Document
                uploaded_file = st.file_uploader(
                    "Upload PDF or DOCX file",
                    type=["pdf", "docx"]
                )
                if uploaded_file and st.button("Start Quiz"):
                    content = self.extract_text_from_file(uploaded_file)
                    if content:
                        st.session_state.questions = self.generate_quiz(content, is_file=True)
                        st.session_state.start_time = datetime.now()
                        st.session_state.quiz_active = True
                        st.rerun()
        else:  # Quiz is active
            # Display timer at the top
            time_up = self.display_timer()
            if time_up:
                self.show_results()
                return

            # Display all questions
            st.write("### Answer all questions and click Submit when done")
            for q in st.session_state.questions:
                with st.container():
                    st.write(f"**Q{q['id']+1}** ({q['marks']} marks) [{q['category']}]: {q['question']}")
                    key = f"q_{q['id']}"
                    st.session_state.answers[key] = st.radio(
                        f"Select answer for question {q['id']+1}:",
                        q['options'],
                        key=key,
                        index=None
                    )

            # Submit button at the end
            if st.button("Submit Quiz"):
                self.show_results()
    
    def calculate_score(self):
        """Calculate final score with marks weightage"""
        total_score = 0
        max_score = 0
        
        for q in st.session_state.questions:
            if not q['is_extra']:  # Only count main questions for max score
                max_score += q['marks']
            
            key = f"q_{q['id']}"
            if key in st.session_state.answers and st.session_state.answers[key] is not None:
                selected_index = q['options'].index(st.session_state.answers[key])
                if selected_index == q['correct_index']:
                    total_score += q['marks']
        
        return total_score, max_score
    
    def show_results(self):
        """Display comprehensive quiz results"""
        st.title("Quiz Results")
        
        total_score, max_score = self.calculate_score()
        score_percentage = (total_score / max_score) * 100 if max_score > 0 else 0
        
        st.write(f"### Final Score: {total_score}/{max_score} ({score_percentage:.1f}%)")
        
        # Category-wise analysis
        categories = {}
        for q in st.session_state.questions:
            cat = q['category']
            if cat not in categories:
                categories[cat] = {"total": 0, "correct": 0, "questions": 0}
            
            if not q['is_extra']:
                categories[cat]["questions"] += 1
                categories[cat]["total"] += q['marks']
                
                key = f"q_{q['id']}"
                if key in st.session_state.answers and st.session_state.answers[key] is not None:
                    selected_index = q['options'].index(st.session_state.answers[key])
                    if selected_index == q['correct_index']:
                        categories[cat]["correct"] += q['marks']
        
        st.write("### Category-wise Performance")
        for cat, data in categories.items():
            if data["total"] > 0:
                cat_percentage = (data["correct"] / data["total"]) * 100
                cat_name = ""
                if cat == "U":
                    cat_name = "Understanding"
                elif cat == "R":
                    cat_name = "Reasoning"
                elif cat == "A":
                    cat_name = "Application"
                st.write(f"- {cat} ({cat_name}): {cat_percentage:.1f}%")
        
        # Pass/Fail status
        passing_score = st.session_state.settings["passing_score"]
        if score_percentage >= passing_score:
            st.success(f"🎉 Congratulations! You passed with {score_percentage:.1f}%")
        else:
            st.error(f"❌ You need {passing_score}% to pass. Your score: {score_percentage:.1f}%")
        
        if st.button("Start New Quiz"):
            # Reset all necessary session state
            st.session_state.quiz_active = False
            st.session_state.questions = []
            st.session_state.answers = {}
            st.session_state.start_time = None
            st.rerun()

# Run the app
if __name__ == "__main__":
    quiz_generator = EnhancedQuizGenerator()
    quiz_generator.run_quiz()
