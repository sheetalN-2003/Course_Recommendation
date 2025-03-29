import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
import os
import speech_recognition as sr
import pyttsx3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_chat import message

# Ensure spaCy model is available
model_name = "en_core_web_sm"
try:
    nlp = spacy.load(model_name)
except OSError:
    os.system(f"python -m spacy download {model_name}")
    nlp = spacy.load(model_name)

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    doc = nlp(text)
    words = [token.lemma_ for token in doc]
    return " ".join(words)

# Load course data
def load_data():
    if 'course_data' not in st.session_state:
        st.session_state['course_data'] = pd.DataFrame(columns=['Course Name', 'Difficulty Level', 'Course Description', 'Skills', 'Ratings', 'Course URL'])

    uploaded_file = st.file_uploader("Upload your course dataset (CSV format)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_columns = ['Course Name', 'Difficulty Level', 'Course Description', 'Skills', 'Ratings', 'Course URL']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.warning(f"Missing columns: {', '.join(missing_columns)}")
            return

        df.fillna('', inplace=True)
        df['tags'] = (df['Course Name'] + ' ' + df['Difficulty Level'] + ' ' + df['Course Description'] + ' ' + df['Skills']).apply(preprocess_text)
        st.session_state['course_data'] = df

    return st.session_state['course_data']

# Course recommendation
def recommend_courses(df, query, num_recommendations=5):
    query_processed = preprocess_text(query)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = vectorizer.fit_transform(df['tags']).toarray()
    query_vector = vectorizer.transform([query_processed]).toarray()
    similarities = cosine_similarity(query_vector, vectors).flatten()
    similar_courses = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:num_recommendations]

    recommendations = []
    for i, score in similar_courses:
        recommendations.append({
            'Course Name': df.iloc[i]['Course Name'],
            'Rating': df.iloc[i]['Ratings'],
            'URL': df.iloc[i]['Course URL'],
            'Similarity Score': round(score, 2)
        })
    return recommendations

# Chatbot interface
def chatbot_interface(df):
    st.subheader("Course Recommendation Chatbot")
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        message(msg['content'], is_user=msg['is_user'])

    user_input = st.text_input("Ask about courses:")
    if st.button("Send") and user_input:
        st.session_state.messages.append({"content": user_input, "is_user": True})
        recommendations = recommend_courses(df, user_input)

        if recommendations:
            bot_response = "Here are some recommended courses:\n"
            for rec in recommendations:
                bot_response += f"- [{rec['Course Name']}]({rec['URL']}) (Rating: {rec['Rating']}, Similarity: {rec['Similarity Score']})\n"
        else:
            bot_response = "No matching courses found."

        st.session_state.messages.append({"content": bot_response, "is_user": False})
        message(bot_response, is_user=False)

# Search bar for courses
def search_bar(df):
    st.subheader("Search Courses")
    search_query = st.text_input("Enter a keyword or skill:")
    if st.button("Search"):
        results = df[df['tags'].str.contains(preprocess_text(search_query), case=False)]
        if not results.empty:
            for _, row in results.iterrows():
                st.markdown(f"- [{row['Course Name']}]({row['Course URL']}) (Rating: {row['Ratings']})")
        else:
            st.write("No courses found.")

# Voice assistant for queries
def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand."
        except sr.RequestError:
            return "Could not connect to Google Speech API."

# Authentication System
def authentication():
    if 'users' not in st.session_state:
        st.session_state['users'] = []
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = None
    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False

# Login / Logout Functions
def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        users = st.session_state.get('users', [])
        if any(user['username'] == username and user['password'] == password for user in users):
            st.session_state['logged_in'] = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid credentials.")

def logout():
    if st.button("Logout"):
        st.session_state['logged_in'] = None
        st.session_state['admin_logged_in'] = False
        st.success("Logged out.")

def register():
    st.subheader("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if password == confirm_password:
            if any(user['username'] == username for user in st.session_state['users']):
                st.error("Username already exists.")
            else:
                st.session_state['users'].append({"username": username, "password": password})
                st.success("Registration successful!")

# Admin Panel
def admin_panel():
    if st.session_state.get('admin_logged_in'):
        st.subheader("Admin Dashboard")
        users = st.session_state.get('users', [])
        if users:
            st.write("### Registered Users")
            for user in users:
                st.write(f"- {user['username']}")
        else:
            st.write("No users found.")
        logout()
    else:
        st.error("Access Denied!")

# Main Streamlit App
def main():
    st.set_page_config(page_title="AI Course Recommendation", layout="wide")
    authentication()
    
    menu = ["Home", "Register", "Login", "Admin", "User"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.title("AI Course Recommendation System")
        st.write("Discover the best courses tailored to your interests.")

    elif choice == "Register":
        register()

    elif choice == "Login":
        login()

    elif choice == "Admin":
        admin_panel()

    elif choice == "User":
        if st.session_state.get('logged_in'):
            st.subheader(f"Welcome, {st.session_state['logged_in']}")
            df = load_data()
            if not df.empty:
                search_bar(df)
                chatbot_interface(df)
        else:
            st.error("Please log in to access.")

if __name__ == '__main__':
    main()
