import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import random

nltk.download('stopwords')

# Preprocessing and stemming function
def preprocess_and_stem(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = [ps.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

# Load and preprocess data
def load_data():
    uploaded_file = st.file_uploader("Upload your course dataset (CSV format)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Handle missing columns dynamically
        required_columns = ['Course Name', 'Difficulty Level', 'Course Description', 'Skills', 'URL']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''

        df.fillna('', inplace=True)

        # Create 'tags' column
        df['tags'] = (
            df['Course Name'] + ' ' + df['Difficulty Level'] + ' ' +
            df['Course Description'] + ' ' + df['Skills']
        )
        df['tags'] = df['tags'].apply(preprocess_and_stem)

        return df
    else:
        return None

# Recommendation function
def recommend_course(df, similarity, course_name):
    course_index = df[df['Course Name'] == course_name].index
    if len(course_index) > 0:
        course_index = course_index[0]
        distances = similarity[course_index]
        course_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommendations = []
        for i in course_list:
            recommendations.append({
                'name': df.iloc[i[0]]['Course Name'],
                'url': df.iloc[i[0]]['URL']
            })
        return recommendations
    else:
        return []

# Chatbot interaction
def chatbot_interaction(df, similarity):
    user_query = st.text_input("Chat with the CourseBot (e.g., 'I want to learn data science'):")

    if st.button("Ask CourseBot"):
        if user_query.strip() == "":
            st.warning("Please enter a query.")
        else:
            processed_query = preprocess_and_stem(user_query)
            cv = CountVectorizer(max_features=5000, stop_words='english')
            vectors = cv.fit_transform(df['tags'].append(pd.Series(processed_query))).toarray()

            query_vector = vectors[-1]  # Extract the last vector (user query)
            query_similarity = cosine_similarity([query_vector], vectors[:-1])

            query_scores = list(enumerate(query_similarity[0]))
            query_scores = sorted(query_scores, key=lambda x: x[1], reverse=True)[:5]

            st.subheader("CourseBot Recommendations")
            for idx, score in query_scores:
                st.write(f"- [{df.iloc[idx]['Course Name']}]({df.iloc[idx]['URL']})")

# Main Streamlit app function
def main():
    st.title("Advanced Course Recommendation System")
    st.write("Upload a course dataset, explore recommendations, and chat with our CourseBot.")

    # Load data
    df = load_data()

    if df is not None:
        # Display dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Vectorize the tags column
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(df['tags']).toarray()

        # Compute cosine similarity
        similarity = cosine_similarity(vectors)

        # Input for course recommendation
        st.subheader("Find Similar Courses")
        course_name = st.text_input("Enter the course name for recommendations:")

        if st.button("Recommend"):
            if course_name.strip() == "":
                st.warning("Please enter a course name.")
            else:
                recommendations = recommend_course(df, similarity, course_name)
                if recommendations:
                    st.subheader("Recommended Courses")
                    for rec in recommendations:
                        st.write(f"- [{rec['name']}]({rec['url']})")
                else:
                    st.warning("No recommendations found for the given course name.")

        # Chatbot feature
        st.subheader("Chat with CourseBot")
        chatbot_interaction(df, similarity)
    else:
        st.info("Please upload a valid dataset to proceed.")

# Run the app
if __name__ == '__main__':
    main()  
