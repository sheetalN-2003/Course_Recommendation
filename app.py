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

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Function to load course data
def load_data():
    if 'course_data' not in st.session_state:
        st.session_state['course_data'] = pd.DataFrame(columns=[
            'Course Name', 'Difficulty Level', 'Course Description', 'Skills', 'Ratings', 'Course URL'
        ])
    uploaded_file = st.file_uploader("Upload Course Dataset (CSV)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.fillna('', inplace=True)
        df['tags'] = df[['Course Name', 'Difficulty Level', 'Course Description', 'Skills']].apply(lambda x: ' '.join(x), axis=1)
        df['tags'] = df['tags'].apply(preprocess_text)
        st.session_state['course_data'] = df
    return st.session_state['course_data']

# Function to recommend courses
def recommend_courses(df, query, num_recommendations=5):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    df['tags'] = df['tags'].apply(preprocess_text)
    vectors = vectorizer.fit_transform(df['tags']).toarray()
    query_vector = vectorizer.transform([preprocess_text(query)]).toarray()
    similarities = cosine_similarity(query_vector, vectors).flatten()
    similar_courses = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:num_recommendations]
    recommendations = [{
        'Course Name': df.iloc[i]['Course Name'],
        'Rating': df.iloc[i]['Ratings'],
        'URL': df.iloc[i]['Course URL'],
        'Similarity Score': round(score, 2)
    } for i, score in similar_courses]
    return recommendations

# Quiz for personalized course recommendations
def quiz_recommendation(df):
    st.subheader("Course Recommendation Quiz")
    interests = st.multiselect("Select your areas of interest:", df['Skills'].explode().unique())
    difficulty = st.selectbox("Preferred difficulty level:", ['Beginner', 'Intermediate', 'Advanced'])
    if st.button("Get Recommendations"):
        filtered_df = df[df['Skills'].apply(lambda x: any(skill in x for skill in interests)) & (df['Difficulty Level'] == difficulty)]
        if not filtered_df.empty:
            st.write("### Recommended Courses:")
            for _, row in filtered_df.iterrows():
                st.markdown(f"- [{row['Course Name']}]({row['Course URL']}) (Rating: {row['Ratings']})")
        else:
            st.write("No matching courses found.")

# Voice Assistant Function
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("mpg321 response.mp3")

# Gamification: Simple quiz-based rewards
def gamification():
    st.subheader("Learning Rewards")
    score = 0
    q1 = st.radio("What is Python mainly used for?", ["Web Development", "Data Science", "Both"])
    if q1 == "Both":
        score += 1
    q2 = st.radio("Which library is used for machine learning?", ["NumPy", "scikit-learn", "Flask"])
    if q2 == "scikit-learn":
        score += 1
    if st.button("Submit Answers"):
        st.write(f"Your score: {score}/2")
        if score == 2:
            st.success("Great job! Here are some advanced courses.")
        else:
            st.warning("Keep learning! Here are some beginner-friendly courses.")

# Chatbot
def chatbot_interface(df):
    st.header("AI Course Recommendation Chatbot")
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        message(msg['content'], is_user=msg['is_user'])
    user_input = st.text_input("You:", key="chat_input", placeholder="Ask about courses...")
    if st.button("Send") and user_input:
        st.session_state.messages.append({"content": user_input, "is_user": True})
        recommendations = recommend_courses(df, user_input)
        bot_response = "Here are some course recommendations:\n"
        for rec in recommendations:
            bot_response += f"- [{rec['Course Name']}]({rec['URL']}) (Rating: {rec['Rating']})\n"
        if not recommendations:
            bot_response = "Sorry, no matching courses found."
        st.session_state.messages.append({"content": bot_response, "is_user": False})
        message(bot_response, is_user=False)
        speak(bot_response)

# Main function
def main():
    st.set_page_config(page_title="AI Course Recommender", layout="wide")
    menu = ["Home", "Register", "Login", "Admin", "User"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.title("Welcome to the AI Course Recommendation System")
    elif choice == "User":
        df = load_data()
        if not df.empty:
            chatbot_interface(df)
            quiz_recommendation(df)
            gamification()
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = None
        st.success("Logged out successfully.")

if __name__ == "__main__":
    main()
