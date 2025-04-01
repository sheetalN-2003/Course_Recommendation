import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import random
import json
import base64
from fpdf import FPDF
import speech_recognition as sr
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import pyotp
import smtplib
from email.mime.text import MIMEText
import requests
from io import BytesIO
from PIL import Image

# Set page config
st.set_page_config(
    page_title="AI Learning Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.users = {
            'admin': {
                'password': 'admin123',
                'role': 'admin',
                'progress': {},
                'preferences': {
                    'learning_style': 'visual',
                    'topics': ['AI', 'Programming'],
                    'difficulty': 'intermediate'
                },
                'joined_date': datetime.now().strftime("%Y-%m-%d"),
                'last_login': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'badges': ['early_adopter'],
                'quiz_scores': {},
                'forum_posts': 0,
                'streak_days': 1,
                'points': 100,
                'oauth': None,
                'totp_secret': None  # No 2FA for admin
            }
        }
        st.session_state.logged_in = None
        st.session_state.courses = pd.DataFrame(columns=[
            'Course Name', 'Description', 'Category', 'Skills', 'Difficulty Level',
            'Duration', 'Price', 'Instructor', 'Ratings', 'Course URL', 'Created At',
            'Syllabus', 'Video Hours', 'Exercises', 'Projects', 'Prerequisites'
        ])
        st.session_state.messages = []
        st.session_state.learning_paths = {}
        st.session_state.forum_posts = []
        st.session_state.quiz_questions = {}
        st.session_state.user_analytics = {}
        st.session_state.code_exercises = {}
        st.session_state.ai_assistant_history = {}
        st.session_state.leaderboard = {}
        st.session_state.certificates = {}
        st.session_state.games = {
            'coding_challenge': {
                'high_scores': {},
                'daily_challenge': None
            }
        }

# Initialize the session state
initialize_session_state()

# ======================
# UTILITY FUNCTIONS
# ======================

def generate_pdf_certificate(user_name, course_name, completion_date):
    """Generate a PDF certificate for course completion"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    
    # Add border
    pdf.rect(5, 5, 200, 287)
    
    # Add logo
    try:
        pdf.image("logo.png", 80, 20, 50)
    except:
        pass
    
    pdf.ln(60)
    pdf.cell(0, 10, "Certificate of Completion", 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font("Arial", '', 18)
    pdf.cell(0, 10, f"This is to certify that", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 22)
    pdf.cell(0, 10, f"{user_name}", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 18)
    pdf.cell(0, 10, "has successfully completed the course", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 10, f"{course_name}", 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font("Arial", '', 16)
    pdf.cell(0, 10, f"Completed on: {completion_date}", 0, 1, 'C')
    
    pdf.ln(30)
    pdf.set_font("Arial", 'I', 14)
    pdf.cell(0, 10, "Certificate ID: " + hashlib.sha256(f"{user_name}{course_name}{completion_date}".encode()).hexdigest()[:16], 0, 1, 'C')
    
    # Save to session state
    cert_id = f"cert_{user_name}_{course_name}"
    st.session_state.certificates[cert_id] = {
        'user': user_name,
        'course': course_name,
        'date': completion_date,
        'pdf': pdf.output(dest='S').encode('latin1')
    }
    
    return cert_id

def create_download_link(val, filename):
    """Generate a download link for files"""
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download Certificate</a>'

def send_email(to, subject, body):
    """Send email (simplified for demo)"""
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = 'noreply@ailearningplatform.com'
        msg['To'] = to
        
        # In production, you would use a real SMTP server
        # with smtplib.SMTP('smtp.server.com', 587) as server:
        #     server.login('user', 'pass')
        #     server.send_message(msg)
        
        st.success(f"Email sent to {to} (simulated)")
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

def generate_totp_secret():
    """Generate a TOTP secret for 2FA"""
    return pyotp.random_base32()

def verify_totp(secret, token):
    """Verify TOTP token"""
    if not secret:
        return False
    totp = pyotp.TOTP(secret)
    return totp.verify(token)

def update_streak(username):
    """Update user login streak"""
    user = st.session_state.users[username]
    last_login = datetime.strptime(user['last_login'], "%Y-%m-%d %H:%M") if 'last_login' in user else None
    today = datetime.now()
    
    if last_login:
        delta = (today - last_login).days
        if delta == 1:  # Consecutive day
            user['streak_days'] += 1
            if user['streak_days'] % 7 == 0:
                award_badge(username, f"streak_{user['streak_days']}_days")
                user['points'] += 50
        elif delta > 1:  # Broken streak
            user['streak_days'] = 1
    else:
        user['streak_days'] = 1
    
    # Award points for daily login
    user['points'] = user.get('points', 0) + 10
    
    # Update leaderboard
    update_leaderboard(username, user['points'])

def award_badge(username, badge_name):
    """Award badge to user"""
    if username in st.session_state.users:
        if 'badges' not in st.session_state.users[username]:
            st.session_state.users[username]['badges'] = []
        
        if badge_name not in st.session_state.users[username]['badges']:
            st.session_state.users[username]['badges'].append(badge_name)
            st.session_state.users[username]['points'] += 20
            return True
    return False

def update_leaderboard(username, points):
    """Update leaderboard with user points"""
    st.session_state.leaderboard[username] = points

def recommend_courses_advanced(df, user_prefs, num_recommendations=5):
    """Advanced course recommendation using NLP and ML techniques"""
    if df.empty:
        return []
    
    # Create feature vectors for each course
    df['feature_vector'] = df['Course Name'] + " " + df['Description'] + " " + \
                          df['Category'] + " " + df['Difficulty Level'] + " " + \
                          df['Skills'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['feature_vector'])
    
    # Create user preference vector
    user_vector = " ".join([
        user_prefs.get('learning_style', ''),
        " ".join(user_prefs.get('topics', [])),
        user_prefs.get('difficulty', '')
    ])
    user_tfidf = tfidf.transform([user_vector])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)
    
    # Get top recommendations
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N courses
    top_indices = [i[0] for i in sim_scores[:num_recommendations]]
    recommendations = df.iloc[top_indices].to_dict('records')
    
    return recommendations

# ======================
# AUTHENTICATION SYSTEM
# ======================

def register():
    st.subheader("Register New User")
    with st.form("register_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_username = st.text_input("Enter Username")
            new_password = st.text_input("Enter Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            email = st.text_input("Email Address")
            
        with col2:
            learning_style = st.selectbox("Preferred Learning Style", 
                                        ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"])
            interests = st.multiselect("Topics of Interest", 
                                     ["AI", "Programming", "Data Science", "Business", "Design", "Math"])
            difficulty = st.selectbox("Preferred Difficulty Level", 
                                    ["Beginner", "Intermediate", "Advanced"])
        
        enable_2fa = st.checkbox("Enable Two-Factor Authentication")
        submitted = st.form_submit_button("Register")
        
        if submitted:
            if new_password != confirm_password:
                st.error("Passwords don't match!")
            elif new_username in st.session_state.users:
                st.warning("Username already exists!")
            else:
                user_data = {
                    'password': new_password,
                    'role': 'user',
                    'progress': {},
                    'preferences': {
                        'learning_style': learning_style.lower(),
                        'topics': interests,
                        'difficulty': difficulty.lower()
                    },
                    'joined_date': datetime.now().strftime("%Y-%m-%d"),
                    'last_login': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'badges': ['new_user'],
                    'quiz_scores': {},
                    'forum_posts': 0,
                    'streak_days': 0,
                    'points': 0,
                    'email': email,
                    'oauth': None
                }
                
                if enable_2fa:
                    secret = generate_totp_secret()
                    user_data['totp_secret'] = secret
                    st.info(f"Scan this QR code with your authenticator app:")
                    
                    # Generate QR code (simulated)
                    totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(name=new_username, issuer_name="AI Learning Platform")
                    st.write(f"Or manually enter this secret: {secret}")
                
                st.session_state.users[new_username] = user_data
                st.success("Registration successful! You can now log in.")
                time.sleep(1)
                st.rerun()

def login():
    st.subheader("Login")
    login_method = st.radio("Login Method", ["Email/Password", "Google OAuth (Simulated)"])
    
    if login_method == "Email/Password":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username in st.session_state.users and st.session_state.users[username]['password'] == password:
                user = st.session_state.users[username]
                
                # Skip 2FA for admin
                if username == 'admin':
                    complete_login(username)
                    return
                
                if 'totp_secret' in user and user['totp_secret']:
                    st.session_state.temp_user = username
                    st.session_state.show_2fa = True
                else:
                    complete_login(username)
            else:
                st.error("Invalid credentials!")
                
        if st.session_state.get('show_2fa'):
            token = st.text_input("Enter 2FA Code")
            if st.button("Verify"):
                if verify_totp(st.session_state.users[st.session_state.temp_user]['totp_secret'], token):
                    complete_login(st.session_state.temp_user)
                    st.session_state.show_2fa = False
                    del st.session_state.temp_user
                else:
                    st.error("Invalid 2FA code")
    else:
        # Simulated Google OAuth
        if st.button("Login with Google (Simulated)"):
            simulated_email = "user@gmail.com"
            username = "google_" + simulated_email.split("@")[0]
            
            if username not in st.session_state.users:
                # Auto-register the user
                st.session_state.users[username] = {
                    'password': None,
                    'role': 'user',
                    'progress': {},
                    'preferences': {
                        'learning_style': 'visual',
                        'topics': ['AI', 'Programming'],
                        'difficulty': 'intermediate'
                    },
                    'joined_date': datetime.now().strftime("%Y-%m-%d"),
                    'last_login': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'badges': ['oauth_user'],
                    'quiz_scores': {},
                    'forum_posts': 0,
                    'streak_days': 1,
                    'points': 50,
                    'email': simulated_email,
                    'oauth': 'google',
                    'totp_secret': None
                }
            
            complete_login(username)

def complete_login(username):
    st.session_state.logged_in = username
    st.session_state.users[username]['last_login'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    update_streak(username)
    st.success(f"Welcome back, {username}!")
    time.sleep(1)
    st.rerun()

def logout():
    st.session_state.logged_in = None
    st.rerun()

# ======================
# COURSE MANAGEMENT
# ======================

def course_marketplace():
    st.subheader("Course Marketplace")
    
    if st.session_state.courses.empty:
        st.warning("No courses available yet.")
        return
    
    for _, course in st.session_state.courses.iterrows():
        with st.expander(f"{course['Course Name']} - {course['Difficulty Level']}"):
            st.write(f"**Description:** {course['Description']}")
            st.write(f"**Category:** {course['Category']}")
            st.write(f"**Duration:** {course['Duration']} hours")
            st.write(f"**Instructor:** {course['Instructor']}")
            st.write(f"**Rating:** {course['Ratings']}/5")
            
            if st.session_state.logged_in:
                if st.button(f"Enroll in {course['Course Name']}", key=f"enroll_{course['Course Name']}"):
                    user = st.session_state.users[st.session_state.logged_in]
                    if course['Course Name'] not in user['progress']:
                        user['progress'][course['Course Name'] = 0
                        st.success("Enrolled successfully!")
                        st.rerun()
                    else:
                        st.warning("You're already enrolled in this course!")
            else:
                st.warning("Please login to enroll in courses")

def admin_add_course():
    st.subheader("Add New Course")
    with st.form("add_course_form"):
        course_data = {
            'Course Name': st.text_input("Course Name"),
            'Description': st.text_area("Description"),
            'Category': st.selectbox("Category", ["Programming", "Data Science", "AI", "Business", "Design"]),
            'Skills': st.multiselect("Skills Covered", ["Python", "Machine Learning", "Data Analysis", "Web Development", "SQL"]),
            'Difficulty Level': st.selectbox("Difficulty Level", ["Beginner", "Intermediate", "Advanced"]),
            'Duration': st.number_input("Duration (hours)", min_value=1, value=10),
            'Price': st.number_input("Price", min_value=0, value=0),
            'Instructor': st.text_input("Instructor Name", value="AI Learning Platform"),
            'Ratings': st.slider("Rating", 1.0, 5.0, 4.5, 0.1),
            'Course URL': st.text_input("Course URL", value="https://example.com/course"),
            'Created At': datetime.now().strftime("%Y-%m-%d"),
            'Syllabus': st.text_area("Syllabus", value="Week 1: Introduction\nWeek 2: Fundamentals\nWeek 3: Advanced Topics"),
            'Video Hours': st.number_input("Video Hours", min_value=0, value=5),
            'Exercises': st.number_input("Number of Exercises", min_value=0, value=10),
            'Projects': st.number_input("Number of Projects", min_value=0, value=2),
            'Prerequisites': st.text_input("Prerequisites", value="None")
        }
        
        if st.form_submit_button("Add Course"):
            new_course = pd.DataFrame([course_data])
            st.session_state.courses = pd.concat([st.session_state.courses, new_course], ignore_index=True)
            st.success("Course added successfully!")
            time.sleep(1)
            st.rerun()

def admin_manage_courses():
    st.subheader("Manage Courses")
    
    if st.session_state.courses.empty:
        st.warning("No courses to manage")
        return
    
    edited_df = st.data_editor(st.session_state.courses, num_rows="dynamic")
    
    if st.button("Save Changes"):
        st.session_state.courses = edited_df
        st.success("Courses updated successfully!")
        time.sleep(1)
        st.rerun()
        
    if st.button("Reset to Original"):
        st.rerun()

# ======================
# LEARNING PATHS
# ======================

def create_learning_path():
    st.subheader("Create Learning Path")
    
    with st.form("learning_path_form"):
        name = st.text_input("Path Name")
        goal = st.text_area("Learning Goal")
        courses = st.multiselect("Select Courses", st.session_state.courses['Course Name'].tolist())
        
        if st.form_submit_button("Create Path"):
            if not name or not goal or not courses:
                st.error("Please fill all fields")
            else:
                path_id = f"path_{len(st.session_state.learning_paths)+1}"
                st.session_state.learning_paths[path_id] = {
                    'name': name,
                    'goal': goal,
                    'courses': courses,
                    'progress': 0,
                    'created_by': st.session_state.logged_in,
                    'created_at': datetime.now().strftime("%Y-%m-%d")
                }
                st.success("Learning path created successfully!")
                time.sleep(1)
                st.rerun()

def view_learning_paths():
    st.subheader("Your Learning Paths")
    
    if not st.session_state.learning_paths:
        st.warning("You haven't created any learning paths yet")
        return
    
    user_paths = [p for p in st.session_state.learning_paths.values() 
                 if p['created_by'] == st.session_state.logged_in]
    
    if not user_paths:
        st.warning("You haven't created any learning paths yet")
        return
    
    for path_id, path in st.session_state.learning_paths.items():
        if path['created_by'] == st.session_state.logged_in:
            with st.expander(f"{path['name']} - {path['progress']}% complete"):
                st.write(f"**Goal:** {path['goal']}")
                st.write("**Courses:**")
                
                total_progress = 0
                for course in path['courses']:
                    course_progress = st.session_state.users[st.session_state.logged_in]['progress'].get(course, 0)
                    st.write(f"- {course} ({course_progress}%)")
                    total_progress += course_progress
                
                # Calculate overall progress
                if path['courses']:
                    overall_progress = total_progress // len(path['courses'])
                    path['progress'] = overall_progress
                
                # Mark as complete button
                if st.button(f"Mark as Completed", key=f"complete_{path_id}"):
                    if path['progress'] < 100:
                        st.warning("Complete all courses to finish this path")
                    else:
                        path['progress'] = 100
                        
                        # Generate certificate
                        cert_id = generate_pdf_certificate(
                            st.session_state.logged_in,
                            path['name'],
                            datetime.now().strftime("%Y-%m-%d")
                        )
                        
                        # Award badge
                        award_badge(st.session_state.logged_in, f"path_completed_{path_id}")
                        
                        st.success("Learning path completed!")
                        st.markdown(create_download_link(
                            st.session_state.certificates[cert_id]['pdf'],
                            f"Certificate_{path['name']}"
                        ), unsafe_allow_html=True)
                        st.rerun()

# ======================
# USER PROGRESS
# ======================

def user_progress():
    st.subheader("Your Learning Progress")
    
    if not st.session_state.logged_in:
        st.warning("Please login to view your progress")
        return
    
    user = st.session_state.users[st.session_state.logged_in]
    
    if not user.get('progress'):
        st.warning("You haven't enrolled in any courses yet")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Course Progress")
        for course, progress in user['progress'].items():
            st.write(f"**{course}**")
            st.progress(progress)
            st.write(f"{progress}% Complete")
            
            # Add progress slider
            new_progress = st.slider(f"Update progress for {course}", 0, 100, progress, key=f"progress_{course}")
            if new_progress != progress:
                user['progress'][course] = new_progress
                st.success("Progress updated!")
                st.rerun()
    
    with col2:
        st.write("### Your Certificates")
        user_certs = [c for c in st.session_state.certificates.values() 
                     if c['user'] == st.session_state.logged_in]
        
        if user_certs:
            for cert in user_certs:
                with st.expander(f"Certificate: {cert['course']}"):
                    st.write(f"Completed on: {cert['date']}")
                    st.markdown(create_download_link(
                        cert['pdf'],
                        f"Certificate_{cert['course']}"
                    ), unsafe_allow_html=True)
        else:
            st.write("You haven't earned any certificates yet.")

# ======================
# AI CHATBOT
# ======================

def ai_chatbot():
    st.subheader("AI Learning Assistant")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Voice input (simulated)
    if st.button("🎤 Use Voice Input (Simulated)"):
        with st.spinner("Listening (simulated)..."):
            time.sleep(2)
            sample_responses = [
                "I need help with Python lists",
                "What courses do you recommend for data science?",
                "How do I reset my password?",
                "Explain machine learning to me"
            ]
            user_input = random.choice(sample_responses)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.rerun()
    
    # Text input
    user_input = st.chat_input("Ask your learning question...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Generate AI response (simulated)
        ai_response = generate_ai_response(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        st.rerun()

def generate_ai_response(query):
    """Generate AI response to user query (simulated with rules)"""
    query = query.lower()
    
    # Course-related questions
    if "course" in query or "learn" in query or "study" in query:
        if "python" in query:
            return "I recommend starting with our 'Python for Beginners' course. It covers all the fundamentals with hands-on exercises."
        elif "data science" in query:
            return "Our 'Data Science Foundations' path includes 3 courses that take you from basics to advanced topics."
        elif "recommend" in query:
            if st.session_state.logged_in and st.session_state.logged_in != "guest":
                user_prefs = st.session_state.users[st.session_state.logged_in].get('preferences', {})
                recs = recommend_courses_advanced(st.session_state.courses, user_prefs, 2)
                if recs:
                    return f"Based on your preferences, I recommend: 1) {recs[0]['Course Name']} and 2) {recs[1]['Course Name']}"
            return "Our most popular courses are 'AI Fundamentals' and 'Web Development Bootcamp'."
    
    # Platform questions
    elif "progress" in query or "track" in query:
        return "You can view your learning progress in the 'My Progress' section. It shows completed courses and your current streak."
    
    # Technical questions
    elif "error" in query or "problem" in query:
        return "Try checking our documentation or forum for similar issues. Would you like me to search the forum for you?"
    
    # Default response
    return "I'm here to help with your learning journey! You can ask me about courses, your progress, or specific topics you're studying."

# ======================
# QUIZ SYSTEM
# ======================

def quiz_system():
    st.subheader("Knowledge Check")
    
    if 'current_quiz' not in st.session_state:
        st.session_state.current_quiz = None
        st.session_state.quiz_answers = {}
        st.session_state.quiz_score = 0
    
    # Initialize sample quizzes if empty
    if not st.session_state.quiz_questions:
        st.session_state.quiz_questions = {
            'Python Basics': {
                'description': 'Test your Python fundamentals',
                'questions': [
                    {
                        'question': 'What is the output of print(2**3)?',
                        'type': 'mcq',
                        'options': ['6', '8', '9', '23'],
                        'answer': '8'
                    },
                    {
                        'question': 'Python is an interpreted language',
                        'type': 'true_false',
                        'answer': 'True'
                    },
                    {
                        'question': 'What keyword defines a function in Python?',
                        'type': 'short_answer',
                        'answer': 'def'
                    }
                ]
            },
            'Data Science Concepts': {
                'description': 'Basic data science knowledge check',
                'questions': [
                    {
                        'question': 'Which library is primarily used for data manipulation in Python?',
                        'type': 'mcq',
                        'options': ['NumPy', 'Pandas', 'Matplotlib', 'Scikit-learn'],
                        'answer': 'Pandas'
                    },
                    {
                        'question': 'Supervised learning requires labeled data',
                        'type': 'true_false',
                        'answer': 'True'
                    }
                ]
            }
        }
    
    # Select quiz
    quiz_options = list(st.session_state.quiz_questions.keys())
    selected_quiz = st.selectbox("Select a Quiz", ["-- Select Quiz --"] + quiz_options)
    
    if selected_quiz != "-- Select Quiz --" and st.button("Start Quiz"):
        st.session_state.current_quiz = selected_quiz
        st.session_state.quiz_answers = {}
        st.session_state.quiz_score = 0
        st.rerun()
    
    # Display quiz
    if st.session_state.current_quiz:
        quiz = st.session_state.quiz_questions[st.session_state.current_quiz]
        
        st.write(f"## {st.session_state.current_quiz}")
        st.write(quiz['description'])
        st.write(f"Questions: {len(quiz['questions'])}")
        
        with st.form("quiz_form"):
            for i, question in enumerate(quiz['questions']):
                st.write(f"**{i+1}. {question['question']}**")
                
                if question['type'] == 'mcq':
                    options = question['options']
                    selected = st.radio(f"Select answer for Q{i+1}", options, key=f"q_{i}")
                    st.session_state.quiz_answers[i] = selected
                elif question['type'] == 'true_false':
                    selected = st.radio(f"True or False for Q{i+1}", ["True", "False"], key=f"q_{i}")
                    st.session_state.quiz_answers[i] = selected
                elif question['type'] == 'short_answer':
                    answer = st.text_input(f"Your answer for Q{i+1}", key=f"q_{i}")
                    st.session_state.quiz_answers[i] = answer
                
                st.write("---")
            
            submitted = st.form_submit_button("Submit Quiz")
            
            if submitted:
                correct = 0
                results = []
                
                for i, question in enumerate(quiz['questions']):
                    user_answer = st.session_state.quiz_answers.get(i, "")
                    correct_answer = question['answer']
                    
                    # For short answers, use similarity
                    if question['type'] == 'short_answer':
                        similarity = SequenceMatcher(None, user_answer.lower(), correct_answer.lower()).ratio()
                        is_correct = similarity > 0.7
                    else:
                        is_correct = str(user_answer).lower() == str(correct_answer).lower()
                    
                    if is_correct:
                        correct += 1
                    
                    results.append({
                        'question': question['question'],
                        'user_answer': user_answer,
                        'correct_answer': correct_answer,
                        'is_correct': is_correct
                    })
                
                # Calculate score
                score = int((correct / len(quiz['questions'])) * 100)
                st.session_state.quiz_score = score
                
                # Store score for logged in users
                if st.session_state.logged_in and st.session_state.logged_in != "guest":
                    user = st.session_state.users[st.session_state.logged_in]
                    user['quiz_scores'][st.session_state.current_quiz] = {
                        'score': score,
                        'date': datetime.now().strftime("%Y-%m-%d")
                    }
                    
                    # Award badge for high scores
                    if score >= 90:
                        award_badge(st.session_state.logged_in, f"quiz_master_{st.session_state.current_quiz.lower().replace(' ', '_')}")
                
                # Display results
                st.write(f"## Your Score: {score}%")
                
                for result in results:
                    st.write(f"**Q:** {result['question']}")
                    st.write(f"**Your Answer:** {result['user_answer']}")
                    st.write(f"**Correct Answer:** {result['correct_answer']}")
                    st.write(f"**Result:** {'✅ Correct' if result['is_correct'] else '❌ Incorrect'}")
                    st.write("---")
                
                # Add retake button
                if st.button("Retake Quiz"):
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_score = 0
                    st.rerun()

# ======================
# CODING CHALLENGE
# ======================

def coding_challenge_game():
    st.subheader("Daily Coding Challenge")
    
    # Initialize or reset daily challenge
    if 'daily_challenge' not in st.session_state.games['coding_challenge'] or \
       st.session_state.games['coding_challenge']['daily_challenge'] is None or \
       st.session_state.games['coding_challenge']['daily_challenge']['date'] != datetime.now().strftime("%Y-%m-%d"):
        
        challenges = [
            {
                'problem': "Write a function that reverses a string.",
                'solution': "def reverse_string(s): return s[::-1]",
                'test_cases': [
                    ("hello", "olleh"),
                    ("world", "dlrow"),
                    ("", ""),
                    ("12345", "54321")
                ]
            },
            {
                'problem': "Write a function that checks if a number is prime.",
                'solution': "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True",
                'test_cases': [
                    (2, True),
                    (17, True),
                    (1, False),
                    (25, False),
                    (29, True)
                ]
            }
        ]
        
        st.session_state.games['coding_challenge']['daily_challenge'] = {
            'challenge': random.choice(challenges),
            'date': datetime.now().strftime("%Y-%m-%d")
        }
    
    challenge = st.session_state.games['coding_challenge']['daily_challenge']['challenge']
    
    st.write(f"**Today's Challenge:** {challenge['problem']}")
    st.write("**Rules:**")
    st.write("- Write a Python function that solves the problem")
    st.write("- Your solution will be tested against multiple test cases")
    st.write("- Points are awarded based on passing test cases")
    
    with st.expander("Example Test Cases"):
        for i, (input_case, expected) in enumerate(challenge['test_cases']):
            st.write(f"Test Case {i+1}: Input = {input_case}, Expected Output = {expected}")
    
    user_code = st.text_area("Write your solution here:", height=200,
                           value="def solution(input):\n    # Your code here\n    return None")
    
    if st.button("Submit Solution"):
        # Execute user code in a safe manner
        try:
            # Create a dictionary to capture the function
            namespace = {}
            exec(user_code, namespace)
            
            if 'solution' not in namespace:
                st.error("Please define a function named 'solution'")
                return
            
            user_func = namespace['solution']
            
            # Run test cases
            passed = 0
            results = []
            
            for input_case, expected in challenge['test_cases']:
                try:
                    result = user_func(input_case)
                    is_correct = result == expected
                    if is_correct:
                        passed += 1
                    
                    results.append({
                        'input': input_case,
                        'output': result,
                        'expected': expected,
                        'correct': is_correct
                    })
                except Exception as e:
                    results.append({
                        'input': input_case,
                        'output': f"Error: {str(e)}",
                        'expected': expected,
                        'correct': False
                    })
            
            # Calculate score
            score = int((passed / len(challenge['test_cases'])) * 100
            
            # Display results
            st.write("### Test Results")
            for i, result in enumerate(results):
                st.write(f"**Test Case {i+1}:**")
                st.write(f"- Input: {result['input']}")
                st.write(f"- Your Output: {result['output']}")
                st.write(f"- Expected: {result['expected']}")
                st.write(f"- {'✅ Passed' if result['correct'] else '❌ Failed'}")
                st.write("---")
            
                       st.write(f"### Final Score: {score}%")
            
            # Update high scores for logged in users
            if st.session_state.logged_in and st.session_state.logged_in != "guest":
                username = st.session_state.logged_in
                high_scores = st.session_state.games['coding_challenge']['high_scores']
                
                if username not in high_scores or score > high_scores[username]['score']:
                    high_scores[username] = {
                        'score': score,
                        'date': datetime.now().strftime("%Y-%m-%d")
                    }
                
                # Award points
                st.session_state.users[username]['points'] += score // 10
                update_leaderboard(username, st.session_state.users[username]['points'])
                
                # Award badge for perfect score
                if score == 100:
                    award_badge(username, "coding_champion")
                    
                st.success(f"You earned {score // 10} points!")
        
        except Exception as e:
            st.error(f"Error in your code: {str(e)}")

# ======================
# COMMUNITY FORUM
# ======================

def community_forum():
    st.subheader("Community Forum")
    
    tab1, tab2, tab3 = st.tabs(["Browse Discussions", "Start New Topic", "My Posts"])
    
    with tab1:
        st.write("### Recent Discussions")
        
        if not st.session_state.forum_posts:
            st.write("No discussions yet. Be the first to post!")
        else:
            for post in reversed(st.session_state.forum_posts):
                with st.expander(f"{post['title']} by {post['author']} ({post['date']})"):
                    st.write(post['content'])
                    st.write(f"**Tags:** {', '.join(post['tags'])}")
                    st.write(f"**Replies:** {len(post['replies'])}")
                    
                    with st.expander("View Replies"):
                        for reply in post['replies']:
                            st.write(f"**{reply['author']}** ({reply['date']}):")
                            st.write(reply['content'])
                            st.write("---")
                    
                    if st.session_state.logged_in and st.session_state.logged_in != "guest":
                        with st.form(key=f"reply_form_{post['id']}"):
                            reply_content = st.text_area("Your Reply")
                            if st.form_submit_button("Post Reply"):
                                post['replies'].append({
                                    'author': st.session_state.logged_in,
                                    'content': reply_content,
                                    'date': datetime.now().strftime("%Y-%m-%d %H:%M")
                                })
                                st.session_state.users[st.session_state.logged_in]['forum_posts'] += 1
                                st.session_state.users[st.session_state.logged_in]['points'] += 5
                                st.rerun()
    
    with tab2:
        if st.session_state.logged_in and st.session_state.logged_in != "guest":
            with st.form("new_post_form"):
                title = st.text_input("Post Title")
                content = st.text_area("Post Content")
                tags = st.multiselect("Tags", ["Question", "Help", "Discussion", "Feedback", "Technical"])
                
                if st.form_submit_button("Create Post"):
                    if not title or not content:
                        st.error("Title and content are required")
                    else:
                        new_post = {
                            'id': len(st.session_state.forum_posts) + 1,
                            'title': title,
                            'content': content,
                            'author': st.session_state.logged_in,
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                            'tags': tags,
                            'replies': []
                        }
                        st.session_state.forum_posts.append(new_post)
                        st.session_state.users[st.session_state.logged_in]['forum_posts'] += 1
                        st.session_state.users[st.session_state.logged_in]['points'] += 10
                        award_badge(st.session_state.logged_in, "community_contributor")
                        st.success("Post created successfully!")
                        time.sleep(1)
                        st.rerun()
        else:
            st.warning("Please log in to create new posts.")
    
    with tab3:
        if st.session_state.logged_in and st.session_state.logged_in != "guest":
            user_posts = [p for p in st.session_state.forum_posts if p['author'] == st.session_state.logged_in]
            
            if user_posts:
                st.write("### Your Posts")
                for post in user_posts:
                    with st.expander(f"{post['title']} ({post['date']})"):
                        st.write(post['content'])
                        st.write(f"**Replies:** {len(post['replies'])}")
                        if st.button(f"Delete Post", key=f"delete_{post['id']}"):
                            st.session_state.forum_posts = [p for p in st.session_state.forum_posts if p['id'] != post['id']]
                            st.success("Post deleted")
                            st.rerun()
            else:
                st.write("You haven't created any posts yet.")
        else:
            st.warning("Please log in to view your posts.")

# ======================
# ANALYTICS DASHBOARDS
# ======================

def user_analytics():
    st.subheader("My Learning Analytics")
    
    if st.session_state.logged_in and st.session_state.logged_in != "guest":
        user = st.session_state.users[st.session_state.logged_in]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Learning Streak", f"{user.get('streak_days', 0)} days")
        with col2:
            st.metric("Total Points", user.get('points', 0))
        with col3:
            st.metric("Forum Contributions", user.get('forum_posts', 0))
        
        st.write("### Progress Overview")
        
        if user.get('progress'):
            # Course progress chart
            courses = list(user['progress'].keys())
            progress = list(user['progress'].values())
            
            fig = px.bar(
                x=courses,
                y=progress,
                labels={'x': 'Course', 'y': 'Progress %'},
                title="Course Completion Progress"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Time spent estimation
            total_hours = 0
            for course, pct in user['progress'].items():
                if course in st.session_state.courses['Course Name'].values:
                    duration = st.session_state.courses[st.session_state.courses['Course Name'] == course]['Duration'].values[0]
                    total_hours += duration * (pct / 100)
            
            st.write(f"**Estimated Total Learning Time:** {int(total_hours)} hours")
        else:
            st.warning("You haven't enrolled in any courses yet.")
        
        # Quiz performance
        if user.get('quiz_scores'):
            st.write("### Quiz Performance")
            quizzes = list(user['quiz_scores'].keys())
            scores = [q['score'] for q in user['quiz_scores'].values()]
            
            fig = px.line(
                x=quizzes,
                y=scores,
                labels={'x': 'Quiz', 'y': 'Score %'},
                title="Quiz Scores Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Badges earned
        if user.get('badges'):
            st.write("### Your Badges")
            cols = st.columns(4)
            for i, badge in enumerate(user['badges']):
                with cols[i % 4]:
                    st.image("https://via.placeholder.com/100", width=50)
                    st.caption(badge.replace('_', ' ').title())
    else:
        st.warning("Please log in to view your analytics")

def admin_analytics():
    st.subheader("Admin Analytics Dashboard")
    
    if st.session_state.logged_in == "admin":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", len(st.session_state.users))
        with col2:
            st.metric("Total Courses", len(st.session_state.courses))
        with col3:
            st.metric("Forum Posts", len(st.session_state.forum_posts))
        
        # User growth chart
        st.write("### User Growth")
        join_dates = [u['joined_date'] for u in st.session_state.users.values()]
        date_counts = pd.Series(join_dates).value_counts().sort_index()
        
        fig = px.line(
            x=date_counts.index,
            y=date_counts.values,
            labels={'x': 'Date', 'y': 'New Users'},
            title="Daily User Signups"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Course popularity
        st.write("### Course Popularity")
        if not st.session_state.courses.empty:
            enrolled_counts = {}
            for user in st.session_state.users.values():
                for course in user.get('progress', {}).keys():
                    enrolled_counts[course] = enrolled_counts.get(course, 0) + 1
            
            if enrolled_counts:
                fig = px.bar(
                    x=list(enrolled_counts.keys()),
                    y=list(enrolled_counts.values()),
                    labels={'x': 'Course', 'y': 'Enrollments'},
                    title="Course Enrollments"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No enrollments yet.")
        
        # Leaderboard
        st.write("### Top Learners")
        if st.session_state.leaderboard:
            top_users = sorted(st.session_state.leaderboard.items(), key=lambda x: x[1], reverse=True)[:10]
            df = pd.DataFrame(top_users, columns=['User', 'Points'])
            st.dataframe(df)
        else:
            st.write("No leaderboard data yet.")
    else:
        st.warning("Admin access required")

# ======================
# MAIN APPLICATION
# ======================

def main():
    # Initialize sample data if empty
    if st.session_state.courses.empty:
        sample_courses = [
            {
                'Course Name': 'Python for Beginners',
                'Description': 'Learn Python programming from scratch',
                'Category': 'Programming',
                'Skills': ['Python', 'Programming', 'Algorithms'],
                'Difficulty Level': 'Beginner',
                'Duration': 20,
                'Price': 0,
                'Instructor': 'Dr. Smith',
                'Ratings': 4.5,
                'Course URL': 'https://example.com/python',
                'Created At': '2023-01-15',
                'Syllabus': 'Variables, Loops, Functions, OOP',
                'Video Hours': 15,
                'Exercises': 50,
                'Projects': 5,
                'Prerequisites': 'None'
            },
            {
                'Course Name': 'Advanced Data Science',
                'Description': 'Master data science with Python and ML',
                'Category': 'Data Science',
                'Skills': ['Python', 'Machine Learning', 'Pandas', 'NumPy'],
                'Difficulty Level': 'Advanced',
                'Duration': 40,
                'Price': 99,
                'Instructor': 'Dr. Johnson',
                'Ratings': 4.8,
                'Course URL': 'https://example.com/datascience',
                'Created At': '2023-02-20',
                'Syllabus': 'EDA, ML Models, Feature Engineering',
                'Video Hours': 30,
                'Exercises': 80,
                'Projects': 8,
                'Prerequisites': 'Python basics'
            }
        ]
        st.session_state.courses = pd.DataFrame(sample_courses)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    if not st.session_state.logged_in:
        menu = ["Home", "Login", "Register"]
    elif st.session_state.users[st.session_state.logged_in]['role'] == "admin":
        menu = ["Dashboard", "Add Courses", "Manage Courses", "Admin Analytics", "Logout"]
    else:
        menu = [
            "Dashboard", "Browse Courses", "Learning Paths", 
            "My Progress", "AI Assistant", "Quizzes", 
            "Coding Challenge", "Community Forum", "Analytics",
            "Logout"
        ]
    
    choice = st.sidebar.selectbox("Menu", menu)
    
    # Display user info in sidebar
    if st.session_state.logged_in:
        user = st.session_state.users[st.session_state.logged_in]
        st.sidebar.markdown(f"**Logged in as:** {st.session_state.logged_in}")
        st.sidebar.markdown(f"**Role:** {user['role'].title()}")
        st.sidebar.markdown(f"**Points:** {user.get('points', 0)}")
        
        if user.get('badges'):
            with st.sidebar.expander("Your Badges"):
                for badge in user['badges']:
                    st.write(f"• {badge.replace('_', ' ').title()}")
    
    # Main content routing
    if choice == "Home":
        st.title("AI Learning Platform")
        st.write("Welcome to our intelligent course recommendation system!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Key Features:")
            st.write("""
            - Personalized course recommendations using AI
            - Interactive coding challenges and quizzes
            - AI-powered learning assistant with voice support
            - Detailed learning analytics and progress tracking
            - Community forum for peer support
            - Gamification with badges and leaderboard
            """)
        
        with col2:
            st.write("### Getting Started:")
            st.write("""
            1. Create an account or login
            2. Browse courses and enroll
            3. Track your progress
            4. Earn badges and points
            5. Connect with other learners
            """)
        
        if st.button("Browse Courses as Guest"):
            st.session_state.logged_in = "guest"
            st.rerun()
    
    elif choice == "Login":
        login()
    
    elif choice == "Register":
        register()
    
    elif choice == "Dashboard":
        st.title(f"Welcome, {st.session_state.logged_in or 'Guest'}!")
        
        if st.session_state.logged_in and st.session_state.logged_in != "guest":
            user = st.session_state.users[st.session_state.logged_in]
            
            if user['role'] == "admin":
                st.write("### Admin Dashboard")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Courses", len(st.session_state.courses))
                    st.metric("Forum Posts", len(st.session_state.forum_posts))
                with col2:
                    st.metric("Total Users", len(st.session_state.users))
                    st.metric("Active Today", sum(1 for u in st.session_state.users.values() 
                                               if u.get('last_login', '').startswith(datetime.now().strftime("%Y-%m-%d"))))
            else:
                st.write("### Your Learning Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    enrolled = len(user.get('progress', {}))
                    st.metric("Enrolled Courses", enrolled)
                with col2:
                    st.metric("Learning Streak", f"{user.get('streak_days', 0)} days")
                with col3:
                    st.metric("Your Points", user.get('points', 0))
                
                st.write("### Recommended For You")
                if not st.session_state.courses.empty:
                    recs = recommend_courses_advanced(
                        st.session_state.courses, 
                        user.get('preferences', {}),
                        3
                    )
                    for rec in recs:
                        with st.expander(f"⭐ {rec['Ratings']} - {rec['Course Name']}"):
                            st.write(rec['Description'])
                            st.write(f"**Skills:** {', '.join(rec['Skills'])}")
                            if st.button("Enroll", key=f"enroll_{rec['Course Name']}"):
                                user['progress'][rec['Course Name']] = 0
                                st.success("Enrolled successfully!")
                                st.rerun()
    
    elif choice == "Add Courses" and st.session_state.logged_in == "admin":
        admin_add_course()
    
    elif choice == "Manage Courses" and st.session_state.logged_in == "admin":
        admin_manage_courses()
    
    elif choice == "Browse Courses":
        course_marketplace()
    
    elif choice == "Learning Paths" and st.session_state.logged_in and st.session_state.logged_in != "guest":
        tab1, tab2 = st.tabs(["My Paths", "Create New"])
        
        with tab1:
            view_learning_paths()
        
        with tab2:
            create_learning_path()
    
    elif choice == "My Progress" and st.session_state.logged_in and st.session_state.logged_in != "guest":
        user_progress()
    
    elif choice == "AI Assistant":
        ai_chatbot()
    
    elif choice == "Quizzes":
        quiz_system()
    
    elif choice == "Coding Challenge":
        coding_challenge_game()
    
    elif choice == "Community Forum":
        community_forum()
    
    elif choice == "Analytics" and st.session_state.logged_in and st.session_state.logged_in != "guest":
        if st.session_state.users[st.session_state.logged_in]['role'] == "admin":
            admin_analytics()
        else:
            user_analytics()
    
    elif choice == "Admin Analytics" and st.session_state.logged_in == "admin":
        admin_analytics()
    
    elif choice == "Logout":
        logout()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **AI Learning Platform** v3.0  
    © 2023 All Rights Reserved
    """)

if __name__ == "__main__":
    main()
