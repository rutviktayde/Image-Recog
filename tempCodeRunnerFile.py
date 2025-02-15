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