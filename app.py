import asyncio
import torch

# ======================
# WORKAROUNDS & FIXES
# ======================
# Torch workaround for Streamlit source watcher
if hasattr(torch._classes, '__path__'):
    torch._classes.__path__ = []

# Event loop workaround
if not hasattr(asyncio, '_get_running_loop'):
    asyncio._get_running_loop = asyncio.get_running_loop

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import speech_recognition as sr
from gtts import gTTS
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_chat import message
import time
import json
from datetime import datetime
import plotly.express as px
from PIL import Image
import base64
import io
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from streamlit_webrtc import webrtc_streamer

def voice_input():
    webrtc_streamer(key="voice-input")
    # Add processing logic for the audio stream

# Initialize NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sentiment_analyzer = pipeline("sentiment-analysis")

# Initialize session state
def init_session_state():
    if 'users' not in st.session_state:
        st.session_state['users'] = {
            'admin': {'password': 'admin123', 'role': 'admin', 'progress': {}, 'preferences': {}}
        }
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = None
    if 'course_data' not in st.session_state:
        st.session_state['course_data'] = pd.DataFrame()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'learning_paths' not in st.session_state:
        st.session_state.learning_paths = {}
    if 'voice_enabled' not in st.session_state:
        st.session_state.voice_enabled = False
    if 'assistant_active' not in st.session_state:
        st.session_state.assistant_active = False

init_session_state()

# Enhanced text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Authentication functions
def register():
    st.subheader("Register New User")
    with st.form("register_form"):
        new_username = st.text_input("Enter Username")
        new_password = st.text_input("Enter Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        role = st.selectbox("Role", ["learner", "educator"])
        submitted = st.form_submit_button("Register")
        
        if submitted:
            if new_password != confirm_password:
                st.error("Passwords don't match!")
            elif new_username in st.session_state['users']:
                st.warning("Username already exists!")
            else:
                st.session_state['users'][new_username] = {
                    'password': new_password,
                    'role': role,
                    'progress': {},
                    'preferences': {},
                    'joined_date': datetime.now().strftime("%Y-%m-%d")
                }
                st.success("Registration successful! You can now log in.")

def login():
    st.subheader("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username in st.session_state['users'] and st.session_state['users'][username]['password'] == password:
                st.session_state['logged_in'] = username
                st.success(f"Welcome back, {username}!")
                time.sleep(1)
                st.experimental_rerun()
            else:
                st.error("Invalid credentials!")

def logout():
    st.session_state['logged_in'] = None
    st.success("Logged out successfully.")
    time.sleep(1)
    st.experimental_rerun()

# Admin panel with enhanced features
def admin_panel():
    if st.session_state['logged_in'] != 'admin':
        st.error("Access Denied: Admins only!")
        return
    
    st.title("Admin Dashboard")
    
    tab1, tab2, tab3, tab4 = st.tabs(["User Management", "Content Management", "Analytics", "System Settings"])
    
    with tab1:
        st.subheader("User Management")
        users_df = pd.DataFrame.from_dict(st.session_state['users'], orient='index')
        st.dataframe(users_df)
        
        with st.expander("Add New User"):
            with st.form("admin_add_user"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                user_role = st.selectbox("Role", ["admin", "learner", "educator"])
                if st.form_submit_button("Add User"):
                    st.session_state['users'][new_username] = {
                        'password': new_password,
                        'role': user_role,
                        'progress': {},
                        'preferences': {}
                    }
                    st.success("User added successfully!")
    
    with tab2:
        st.subheader("Content Management")
        if not st.session_state['course_data'].empty:
            st.dataframe(st.session_state['course_data'])
        
        with st.expander("Add Course Manually"):
            with st.form("add_course_form"):
                name = st.text_input("Course Name")
                difficulty = st.selectbox("Difficulty Level", ["Beginner", "Intermediate", "Advanced"])
                description = st.text_area("Description")
                skills = st.text_input("Skills (comma separated)")
                rating = st.slider("Rating", 1.0, 5.0, 4.0)
                url = st.text_input("Course URL")
                
                if st.form_submit_button("Add Course"):
                    new_course = {
                        'Course Name': name,
                        'Difficulty Level': difficulty,
                        'Course Description': description,
                        'Skills': skills,
                        'Ratings': rating,
                        'Course URL': url
                    }
                    st.session_state['course_data'] = st.session_state['course_data'].append(new_course, ignore_index=True)
                    st.success("Course added successfully!")
    
    with tab3:
        st.subheader("Analytics Dashboard")
        if not st.session_state['course_data'].empty:
            fig1 = px.histogram(st.session_state['course_data'], x="Difficulty Level", title="Course Difficulty Distribution")
            st.plotly_chart(fig1)
            
            fig2 = px.scatter(st.session_state['course_data'], x="Ratings", y="Difficulty Level", 
                             color="Difficulty Level", title="Ratings by Difficulty Level")
            st.plotly_chart(fig2)
    
    with tab4:
        st.subheader("System Settings")
        st.session_state['voice_enabled'] = st.checkbox("Enable Voice Assistant Globally", value=st.session_state['voice_enabled'])
        st.info("System version: 2.1.0")

# Voice assistant functions
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    st.audio(audio_file, format='audio/mp3')

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None

# Enhanced course recommendation with multiple strategies
def recommend_courses(df, query, num_recommendations=5, strategy="content_based"):
    if df.empty:
        return []
    
    df['tags'] = df[['Course Name', 'Difficulty Level', 'Course Description', 'Skills']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    df['tags'] = df['tags'].apply(preprocess_text)
    
    if strategy == "content_based":
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        vectors = vectorizer.fit_transform(df['tags']).toarray()
        query_vector = vectorizer.transform([preprocess_text(query)]).toarray()
        similarities = cosine_similarity(query_vector, vectors).flatten()
    elif strategy == "popularity_based":
        similarities = df['Ratings'].values / 5.0  # Normalize ratings to 0-1
    elif strategy == "hybrid":
        # Combine content-based and popularity-based
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        vectors = vectorizer.fit_transform(df['tags']).toarray()
        query_vector = vectorizer.transform([preprocess_text(query)]).toarray()
        content_sim = cosine_similarity(query_vector, vectors).flatten()
        popularity = df['Ratings'].values / 5.0
        similarities = 0.7 * content_sim + 0.3 * popularity
    
    similar_courses = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:num_recommendations]
    
    recommendations = [{
        'Course Name': df.iloc[i]['Course Name'],
        'Difficulty': df.iloc[i]['Difficulty Level'],
        'Rating': df.iloc[i]['Ratings'],
        'URL': df.iloc[i]['Course URL'],
        'Similarity Score': round(score, 2),
        'Description': df.iloc[i]['Course Description'],
        'Skills': df.iloc[i]['Skills']
    } for i, score in similar_courses]
    
    return recommendations

# Course scraping function (for demo purposes)
def scrape_course_info(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # This is a simplified example - real implementation would need to be tailored to each platform
        title = soup.find('h1').text if soup.find('h1') else "Unknown"
        description = soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else ""
        
        return {
            'title': title,
            'description': description
        }
    except Exception as e:
        st.error(f"Error scraping course info: {str(e)}")
        return None

# Enhanced chatbot with voice capabilities
def chatbot_interface(df):
    st.header("AI Learning Assistant")
    
    # Voice assistant toggle
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🎙️ Start Voice Assistant"):
            st.session_state['assistant_active'] = True
            user_input = speech_to_text()
            if user_input:
                st.session_state.messages.append({"content": user_input, "is_user": True})
                process_chat_query(df, user_input)
    
    # Display chat messages
    for i, msg in enumerate(st.session_state.messages):
        message(msg['content'], is_user=msg['is_user'], key=f"msg_{i}")
        
        # Add voice for assistant responses
        if not msg['is_user'] and st.session_state['voice_enabled']:
            text_to_speech(msg['content'])
    
    # Text input
    user_input = st.text_input("Type your message here...", key="chat_input")
    if st.button("Send") and user_input:
        st.session_state.messages.append({"content": user_input, "is_user": True})
        process_chat_query(df, user_input)

def process_chat_query(df, query):
    # Analyze sentiment
    sentiment = sentiment_analyzer(query)[0]
    
    # Generate response based on query
    if any(word in query.lower() for word in ["hi", "hello", "hey"]):
        response = f"Hello {st.session_state['logged_in']}! How can I help you with your learning today?"
    elif any(word in query.lower() for word in ["thank", "thanks"]):
        response = "You're welcome! Is there anything else I can help you with?"
    elif any(word in query.lower() for word in ["course", "learn", "study", "recommend"]):
        recommendations = recommend_courses(df, query, strategy="hybrid")
        if recommendations:
            response = "Here are some courses you might find interesting:\n\n"
            for rec in recommendations:
                response += f"📚 **[{rec['Course Name']}]({rec['URL']})**\n"
                response += f"   - Level: {rec['Difficulty']}\n"
                response += f"   - Rating: {rec['Rating']}/5\n"
                response += f"   - Skills: {rec['Skills']}\n\n"
            response += "Would you like more details about any of these?"
        else:
            response = "I couldn't find any courses matching your query. Could you try being more specific?"
    else:
        response = "I'm here to help with course recommendations and learning guidance. Could you tell me more about what you're looking to learn?"
    
    st.session_state.messages.append({"content": response, "is_user": False})

# Learning path creation
def create_learning_path(df):
    st.subheader("Create Personalized Learning Path")
    
    with st.form("learning_path_form"):
        path_name = st.text_input("Learning Path Name")
        goal = st.text_area("Your Learning Goal")
        time_commitment = st.selectbox("Weekly Time Commitment", 
                                     ["<2 hours", "2-5 hours", "5-10 hours", "10+ hours"])
        difficulty = st.select_slider("Preferred Difficulty", 
                                    ["Beginner", "Intermediate", "Advanced"])
        
        if st.form_submit_button("Create Learning Path"):
            # Generate recommendations based on goal
            recommendations = recommend_courses(df, goal, num_recommendations=5)
            
            if recommendations:
                path_id = f"path_{int(time.time())}"
                st.session_state.learning_paths[path_id] = {
                    'name': path_name,
                    'goal': goal,
                    'courses': recommendations,
                    'created_at': datetime.now().strftime("%Y-%m-%d"),
                    'progress': 0
                }
                
                st.success(f"Learning path '{path_name}' created successfully!")
                display_learning_path(path_id)
            else:
                st.warning("Couldn't find suitable courses for this learning path.")

def display_learning_path(path_id):
    path = st.session_state.learning_paths[path_id]
    
    st.subheader(path['name'])
    st.write(f"**Goal:** {path['goal']}")
    st.write(f"**Progress:** {path['progress']}%")
    
    st.write("### Courses in this path:")
    for i, course in enumerate(path['courses'], 1):
        with st.expander(f"{i}. {course['Course Name']}"):
            st.write(f"**Description:** {course['Description']}")
            st.write(f"**Difficulty:** {course['Difficulty']}")
            st.write(f"**Rating:** {course['Rating']}/5")
            st.write(f"[Course Link]({course['URL']})")
            
            if st.button(f"Mark as Completed", key=f"complete_{path_id}_{i}"):
                path['progress'] = min(100, path['progress'] + (100 / len(path['courses'])))
                st.experimental_rerun()

# Enhanced quiz recommendation
def quiz_recommendation(df):
    st.subheader("Personalized Course Finder")
    
    with st.form("quiz_form"):
        st.write("### Tell us about your learning preferences")
        
        col1, col2 = st.columns(2)
        with col1:
            interests = st.multiselect("Select your areas of interest:", 
                                      ["Data Science", "Web Development", "AI", "Business", "Design"])
            experience = st.selectbox("Current skill level:", 
                                    ["Beginner", "Intermediate", "Advanced"])
        
        with col2:
            learning_style = st.selectbox("Preferred learning style:", 
                                        ["Video Lectures", "Interactive", "Reading", "Hands-on Projects"])
            time_commitment = st.selectbox("Weekly study time:", 
                                         ["<2 hours", "2-5 hours", "5-10 hours", "10+ hours"])
        
        if st.form_submit_button("Find My Courses"):
            # Build query based on responses
            query = f"{' '.join(interests)} courses for {experience} level with {learning_style} style"
            
            # Get recommendations
            recommendations = recommend_courses(df, query, num_recommendations=8, strategy="hybrid")
            
            if recommendations:
                st.write("### Your Personalized Course Recommendations")
                
                # Filter by learning style
                if learning_style == "Video Lectures":
                    recommendations = [r for r in recommendations if "video" in r['Description'].lower()]
                elif learning_style == "Interactive":
                    recommendations = [r for r in recommendations if "interactive" in r['Description'].lower()]
                
                # Display as cards
                cols = st.columns(2)
                for i, rec in enumerate(recommendations[:4]):
                    with cols[i % 2]:
                        with st.container():
                            st.markdown(f"#### [{rec['Course Name']}]({rec['URL']})")
                            st.write(f"**Level:** {rec['Difficulty']}")
                            st.write(f"**Rating:** ⭐ {rec['Rating']}/5")
                            st.progress(int(float(rec['Rating']) / 5 * 100))
                            with st.expander("Details"):
                                st.write(rec['Description'])
            else:
                st.warning("Couldn't find courses matching your preferences. Try broadening your criteria.")

# Enhanced gamification with badges and progress tracking
def gamification():
    st.subheader("Your Learning Journey")
    
    if st.session_state['logged_in']:
        user = st.session_state['users'][st.session_state['logged_in']]
        
        # Initialize progress if not exists
        if 'progress' not in user:
            user['progress'] = {}
        if 'badges' not in user:
            user['badges'] = []
        
        # Progress tracking
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Courses Started", len(user['progress']))
        with col2:
            completed = sum(1 for p in user['progress'].values() if p == 100)
            st.metric("Courses Completed", completed)
        with col3:
            st.metric("Badges Earned", len(user['badges']))
        
        # Badge system
        st.write("### Your Badges")
        if not user['badges']:
            st.write("No badges yet. Complete courses to earn badges!")
        else:
            cols = st.columns(4)
            for i, badge in enumerate(user['badges']):
                with cols[i % 4]:
                    st.image(f"https://via.placeholder.com/100?text={badge}", width=50)
                    st.caption(badge)
        
        # Suggested actions
        st.write("### Next Steps")
        if completed == 0:
            st.write("🎯 Start your first course to earn the 'Starter' badge!")
        elif completed < 3:
            st.write("🔥 Complete 3 courses to earn the 'Learner' badge!")
        else:
            st.write("🚀 You're on a roll! Consider creating a learning path for your next goal.")
    else:
        st.warning("Please log in to track your learning progress.")

# Course marketplace simulation
def course_marketplace(df):
    st.subheader("Course Marketplace")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        difficulty = st.multiselect("Difficulty Level", 
                                   ["Beginner", "Intermediate", "Advanced"],
                                   default=["Beginner", "Intermediate", "Advanced"])
    with col2:
        min_rating = st.slider("Minimum Rating", 1.0, 5.0, 3.5)
    with col3:
        category = st.selectbox("Category", ["All"] + list(df['Skills'].explode().unique()))
    
    # Apply filters
    filtered_df = df[df['Difficulty Level'].isin(difficulty) & (df['Ratings'] >= min_rating)]
    if category != "All":
        filtered_df = filtered_df[filtered_df['Skills'].str.contains(category, case=False)]
    
    # Display results
    if not filtered_df.empty:
        st.write(f"Showing {len(filtered_df)} courses:")
        
        for _, row in filtered_df.iterrows():
            with st.expander(f"{row['Course Name']} - ⭐ {row['Ratings']}"):
                st.write(f"**Level:** {row['Difficulty Level']}")
                st.write(f"**Skills:** {row['Skills']}")
                st.write(f"**Description:** {row['Course Description']}")
                st.write(f"[Enroll Now]({row['Course URL']})")
                
                # Simulate enrollment
                if st.button("Add to My Learning", key=f"enroll_{row['Course Name']}"):
                    if st.session_state['logged_in']:
                        user = st.session_state['users'][st.session_state['logged_in']]
                        if 'progress' not in user:
                            user['progress'] = {}
                        user['progress'][row['Course Name']] = 0
                        st.success("Course added to your learning list!")
                    else:
                        st.warning("Please log in to enroll in courses")
    else:
        st.write("No courses match your filters. Try adjusting your criteria.")

# Data loading with caching
@st.cache_data(ttl=3600)
def load_data():
    if not st.session_state['course_data'].empty:
        return st.session_state['course_data']
    
    # Try to load sample data if no data uploaded
    try:
        sample_data = pd.read_csv("https://raw.githubusercontent.com/example/course-data/main/sample_courses.csv")
        sample_data['tags'] = sample_data[['Course Name', 'Difficulty Level', 'Course Description', 'Skills']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        sample_data['tags'] = sample_data['tags'].apply(preprocess_text)
        st.session_state['course_data'] = sample_data
        return sample_data
    except:
        return pd.DataFrame()

# Main application
def main():
    st.set_page_config(
        page_title="AI Learning Platform",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load data
    df = load_data()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    if not st.session_state['logged_in']:
        menu = ["Home", "Login", "Register"]
    else:
        if st.session_state['users'][st.session_state['logged_in']]['role'] == "admin":
            menu = ["Dashboard", "Admin Panel", "Marketplace", "Learning Paths", "Chat Assistant", "My Progress", "Logout"]
        else:
            menu = ["Dashboard", "Marketplace", "Learning Paths", "Chat Assistant", "My Progress", "Logout"]
    
    choice = st.sidebar.selectbox("Menu", menu)
    
    # Main content area
    if choice == "Home":
        st.title("AI-Powered Learning Platform")
        st.write("""
        Welcome to our intelligent course recommendation system! 
        This platform helps you discover the best learning resources tailored to your needs.
        """)
        
        if st.button("Explore as Guest"):
            st.session_state['logged_in'] = "guest"
            st.experimental_rerun()
    
    elif choice == "Login":
        login()
    
    elif choice == "Register":
        register()
    
    elif choice == "Dashboard" and st.session_state['logged_in']:
        st.title(f"Welcome, {st.session_state['logged_in']}!")
        
        if not df.empty:
            st.subheader("Recommended For You")
            recommendations = recommend_courses(df, "", num_recommendations=3, strategy="popularity_based")
            
            cols = st.columns(3)
            for i, rec in enumerate(recommendations):
                with cols[i]:
                    with st.container():
                        st.markdown(f"#### [{rec['Course Name']}]({rec['URL']})")
                        st.write(f"**Level:** {rec['Difficulty']}")
                        st.write(f"**Rating:** ⭐ {rec['Rating']}/5")
                        st.write(rec['Description'][:100] + "...")
        
        st.subheader("Quick Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎯 Find Courses"):
                st.experimental_rerun()
        with col2:
            if st.button("📚 Create Learning Path"):
                st.experimental_rerun()
        with col3:
            if st.button("💬 Chat with Assistant"):
                st.experimental_rerun()
    
    elif choice == "Admin Panel" and st.session_state['logged_in'] == "admin":
        admin_panel()
    
    elif choice == "Marketplace" and st.session_state['logged_in']:
        course_marketplace(df)
    
    elif choice == "Learning Paths" and st.session_state['logged_in']:
        tab1, tab2 = st.tabs(["My Paths", "Create New"])
        
        with tab1:
            if st.session_state.learning_paths:
                for path_id in st.session_state.learning_paths:
                    display_learning_path(path_id)
            else:
                st.write("You haven't created any learning paths yet.")
        
        with tab2:
            create_learning_path(df)
    
    elif choice == "Chat Assistant" and st.session_state['logged_in']:
        chatbot_interface(df)
    
    elif choice == "My Progress" and st.session_state['logged_in']:
        gamification()
    
    elif choice == "Logout" and st.session_state['logged_in']:
        logout()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **AI Learning Platform** v2.1  
    © 2023 All Rights Reserved  
    [Privacy Policy] | [Terms of Service]
    """)

if __name__ == "__main__":
    main()
