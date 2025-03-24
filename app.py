import streamlit as st
import pandas as pd
import re
import spacy
import openai
import speech_recognition as sr
import pyttsx3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_chat import message
import random

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize session state for authentication
if 'users' not in st.session_state:
    st.session_state['users'] = []
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = None
if 'admin_logged_in' not in st.session_state:
    st.session_state['admin_logged_in'] = False

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(words)

# Load course data
def load_data():
    if 'course_data' not in st.session_state:
        st.session_state['course_data'] = pd.DataFrame(columns=[
            'Course Name', 'Difficulty Level', 'Course Description', 'Skills', 'Ratings', 'Course URL'
        ])
    uploaded_file = st.file_uploader("Upload your course dataset (CSV format)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.fillna('', inplace=True)
        df['tags'] = df.apply(lambda x: preprocess_text(f"{x['Course Name']} {x['Difficulty Level']} {x['Course Description']} {x['Skills']}"), axis=1)
        st.session_state['course_data'] = df
    return st.session_state['course_data']

# Recommend courses
def recommend_courses(df, query, num_recommendations=5):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    query_processed = preprocess_text(query)
    vectors = vectorizer.fit_transform(df['tags']).toarray()
    query_vector = vectorizer.transform([query_processed]).toarray()
    similarities = cosine_similarity(query_vector, vectors).flatten()
    similar_courses = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:num_recommendations]
    return [df.iloc[i[0]] for i in similar_courses]

# AI Voice Assistant
def voice_assistant():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I didn't understand that."

# Gamification Feature
def mini_quiz():
    st.write("Answer a quick question to get personalized recommendations!")
    options = ["Beginner", "Intermediate", "Advanced"]
    level = st.radio("Select your expertise level:", options)
    interest = st.text_input("Enter a topic you're interested in:")
    if st.button("Get Recommendations"):
        df = st.session_state.get('course_data', pd.DataFrame())
        if not df.empty:
            courses = recommend_courses(df, f"{interest} {level}")
            for course in courses:
                st.markdown(f"- [{course['Course Name']}]({course['Course URL']}) (Rating: {course['Ratings']})")

# Chatbot with Voice Feature
def chatbot_interface():
    st.markdown("### AI Course Chatbot")
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    for msg in st.session_state.messages:
        message(msg['content'], is_user=msg['is_user'])
    user_input = st.text_input("You:")
    if st.button("Ask AI"):
        st.session_state.messages.append({"content": user_input, "is_user": True})
        response = recommend_courses(st.session_state.get('course_data', pd.DataFrame()), user_input)
        bot_response = "Here are some recommendations:\n" + '\n'.join([f"- {course['Course Name']}" for course in response])
        st.session_state.messages.append({"content": bot_response, "is_user": False})
        message(bot_response, is_user=False)
        engine.say(bot_response)
        engine.runAndWait()

# User Authentication
def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if any(user['username'] == username and user['password'] == password for user in st.session_state['users']):
            st.session_state['logged_in'] = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid username or password.")

def register():
    st.subheader("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        st.session_state['users'].append({"username": username, "password": password})
        st.success("Registration successful!")

def logout():
    if st.button("Logout"):
        st.session_state['logged_in'] = None
        st.success("Logged out successfully!")

# Main App
def main():
    st.set_page_config(page_title="Advanced Course Recommender", layout="wide")
    st.title("AI-Powered Course Recommendation System")
    menu = ["Home", "Login", "Register", "Chatbot", "Upload Data", "Mini Quiz"]
    choice = st.sidebar.selectbox("Navigation", menu)
    if choice == "Home":
        st.write("Welcome to the AI-driven Course Recommendation System!")
    elif choice == "Login":
        login()
    elif choice == "Register":
        register()
    elif choice == "Chatbot":
        chatbot_interface()
    elif choice == "Upload Data":
        load_data()
    elif choice == "Mini Quiz":
        mini_quiz()
    if st.session_state['logged_in']:
        logout()

if __name__ == '__main__':
    main()
