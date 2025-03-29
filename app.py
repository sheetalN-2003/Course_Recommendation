import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_chat import message
import speech_recognition as sr
import pyttsx3

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Initialize Speech Engine
engine = pyttsx3.init()

# Load Data
def load_data():
    if 'course_data' not in st.session_state:
        st.session_state['course_data'] = pd.DataFrame(columns=['Course Name', 'Difficulty Level', 'Course Description', 'Skills', 'Ratings', 'Course URL'])

    uploaded_file = st.file_uploader("Upload Course Dataset (CSV)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required_columns = ['Course Name', 'Difficulty Level', 'Course Description', 'Skills', 'Ratings', 'Course URL']
        df.fillna('', inplace=True)
        df['tags'] = (df['Course Name'] + ' ' + df['Difficulty Level'] + ' ' + df['Course Description'] + ' ' + df['Skills']).apply(preprocess_text)
        st.session_state['course_data'] = df

    return st.session_state['course_data']

# Recommend Courses
def recommend_courses(df, query, num_recommendations=5):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    df['tags'] = df['tags'].apply(preprocess_text)
    vectors = vectorizer.fit_transform(df['tags']).toarray()
    query_vector = vectorizer.transform([preprocess_text(query)]).toarray()
    similarities = cosine_similarity(query_vector, vectors).flatten()
    top_courses = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:num_recommendations]
    
    return [{"Course Name": df.iloc[i]['Course Name'], "URL": df.iloc[i]['Course URL'], "Rating": df.iloc[i]['Ratings']} for i, _ in top_courses]

# Search Courses
def search_courses(df):
    query = st.text_input("Enter skill or keyword:")
    if st.button("Search") and query:
        results = df[df['tags'].str.contains(preprocess_text(query), case=False)]
        st.write("### Search Results")
        for _, row in results.iterrows():
            st.markdown(f"- [{row['Course Name']}]({row['Course URL']}) (Rating: {row['Ratings']})")

# Chatbot with Voice Assistant
def chatbot_interface(df):
    st.subheader("AI Course Recommendation Chatbot")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        message(msg['content'], is_user=msg['is_user'])

    user_input = st.text_input("You:", key="chat_input")
    
    if st.button("Send"):
        st.session_state.messages.append({"content": user_input, "is_user": True})
        recommendations = recommend_courses(df, user_input)

        if recommendations:
            bot_response = "Here are some courses for you:\n"
            for rec in recommendations:
                bot_response += f"- [{rec['Course Name']}]({rec['URL']}) (Rating: {rec['Rating']})\n"
        else:
            bot_response = "No matching courses found."

        st.session_state.messages.append({"content": bot_response, "is_user": False})
        engine.say(bot_response)
        engine.runAndWait()
        message(bot_response, is_user=False)

# User Authentication System
def user_auth():
    if 'users' not in st.session_state:
        st.session_state['users'] = []
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = None
    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False

    menu = ["Home", "Register", "Login", "Logout", "Admin Login", "Admin Logout", "User"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.title("Course Recommendation System")
    elif choice == "Register":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            if any(user['username'] == username for user in st.session_state['users']):
                st.error("Username already exists.")
            else:
                st.session_state['users'].append({"username": username, "password": password})
                st.success("Registration successful!")
    elif choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if any(user['username'] == username and user['password'] == password for user in st.session_state['users']):
                st.session_state['logged_in'] = username
                st.success(f"Welcome, {username}!")
            else:
                st.error("Invalid credentials")
    elif choice == "Logout":
        st.session_state['logged_in'] = None
        st.success("You have been logged out.")
    elif choice == "Admin Login":
        admin_username = st.text_input("Admin Username")
        admin_password = st.text_input("Admin Password", type="password")
        if st.button("Login as Admin"):
            if admin_username == "admin" and admin_password == "admin123":
                st.session_state['admin_logged_in'] = True
                st.success("Admin logged in successfully.")
            else:
                st.error("Invalid admin credentials")
    elif choice == "Admin Logout":
        st.session_state['admin_logged_in'] = False
        st.success("Admin logged out.")
    elif choice == "User":
        if st.session_state['logged_in']:
            st.subheader(f"Welcome, {st.session_state['logged_in']}")
            df = load_data()
            if not df.empty:
                search_courses(df)
                chatbot_interface(df)
        else:
            st.error("Please log in.")

# Run Streamlit App
if __name__ == "__main__":
    st.set_page_config(page_title="AI Course Recommendation System", layout="wide")
    user_auth()
