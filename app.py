import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from streamlit_chat import message

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

# Preprocessing and stemming function
def preprocess_and_stem(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    words = [ps.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

def load_data():
    uploaded_file = st.file_uploader("Upload your course dataset (CSV format)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Display columns in the uploaded file for debugging
        st.write("Columns in the uploaded file:", df.columns)

        # Remove any leading or trailing spaces from the column names
        df.columns = df.columns.str.strip()

        # Define the expected columns
        expected_columns = ['Course Name', 'Difficulty Level', 'Course Description', 'Skills', 'Ratings', 'Course URL']
        
        # Check if the required columns are present in the uploaded dataset
        missing_columns = [col for col in expected_columns if col not in df.columns]

        # If there are missing columns, show an error message with the list of missing columns
        if missing_columns:
            st.error(f"The uploaded dataset is missing the following required columns: {', '.join(missing_columns)}")
            return None

        # Proceed with available columns and clean the data
        df = df[expected_columns]
        df.fillna('', inplace=True)

        # Create 'tags' column for recommendations
        df['tags'] = (df['Course Name'].astype(str) + ' ' +
                      df['Difficulty Level'].astype(str) + ' ' +
                      df['Course Description'].astype(str) + ' ' +
                      df['Skills'].astype(str))
        df['tags'] = df['tags'].apply(preprocess_and_stem)

        # Add 'Ratings' column if missing
        if 'Ratings' not in df.columns:
            df['Ratings'] = 'N/A'

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
            course_data = {
                'name': df.iloc[i[0]]['Course Name'],
                'rating': df.iloc[i[0]].get('Ratings', 'N/A'),
                'link': df.iloc[i[0]].get('Course URL', '#')
            }
            recommendations.append(course_data)
        return recommendations
    else:
        return []

# Search function
def search_courses(df, query):
    query = preprocess_and_stem(query)
    results = df[df['tags'].str.contains(query, na=False)]
    return results[['Course Name', 'Ratings', 'Course URL']]

# Registration function
def register_user():
    with st.form("register_form"):
        st.subheader("Register")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        if st.form_submit_button("Register"):
            if password == confirm_password:
                st.session_state['users'] = st.session_state.get('users', [])
                st.session_state['users'].append({'username': username, 'password': password})
                st.success(f"User {username} registered successfully!")
            else:
                st.error("Passwords do not match.")

# Login function
def login_user():
    with st.form("login_form"):
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            users = st.session_state.get('users', [])
            if any(user['username'] == username and user['password'] == password for user in users):
                st.session_state['logged_in'] = username
                st.success(f"Welcome, {username}!")
            else:
                st.error("Invalid username or password.")

# Admin page
def admin_page():
    st.subheader("Admin Page")
    st.write("Manage users and data.")
    if st.button("View Registered Users"):
        users = st.session_state.get('users', [])
        if users:
            st.write(users)
        else:
            st.write("No registered users.")

# User page
def user_page():
    st.subheader("User Page")
    st.write("Welcome, User! Explore courses and get recommendations.")

# Chatbot function
def chatbot_interface(df, similarity):
    st.subheader("Chatbot for Course Recommendations")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        message(msg['content'], is_user=msg['is_user'])

    user_input = st.text_input("You:", key="chat_input")
    if st.button("Send") and user_input:
        st.session_state.messages.append({"content": user_input, "is_user": True})

        # Process chatbot response
        recommendations = recommend_course(df, similarity, user_input)
        if recommendations:
            bot_response = "Here are some course recommendations:\n"
            for rec in recommendations:
                bot_response += f"- [{rec['name']}]({rec['link']}) (Rating: {rec['rating']})\n"
        else:
            bot_response = "Sorry, I couldn't find any courses matching your input."

        st.session_state.messages.append({"content": bot_response, "is_user": False})
        message(bot_response, is_user=False)

# Search bar function
def search_bar(df):
    st.subheader("Search Courses")
    search_query = st.text_input("Search for a course:")
    if st.button("Search") and search_query:
        results = search_courses(df, search_query)
        if not results.empty:
            st.write("### Search Results")
            for _, row in results.iterrows():
                st.markdown(f"- [{row['Course Name']}]({row['Course URL']}) (Rating: {row['Ratings']})")
        else:
            st.write("No courses found matching your query.")

# Main Streamlit app function
def main():
    st.set_page_config(page_title="Course Recommendation System", layout="wide")

    st.title("Course Recommendation System")
    st.write("Upload a course dataset and explore recommendations interactively.")

    # Navigation
    menu = ["Home", "Register", "Login", "Admin", "User"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        # Load data
        df = load_data()
        if df is not None:
            st.session_state['df'] = df

            # Display dataset preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head())

            # Vectorize the tags column
            cv = CountVectorizer(max_features=5000, stop_words='english')
            vectors = cv.fit_transform(df['tags']).toarray()

            # Compute cosine similarity
            similarity = cosine_similarity(vectors)
            st.session_state['similarity'] = similarity

            # Chatbot interface
            chatbot_interface(df, similarity)

            # Search bar
            search_bar(df)
        else:
            st.info("Please upload a valid dataset to proceed.")
    elif choice == "Register":
        register_user()
    elif choice == "Login":
        login_user()
    elif choice == "Admin":
        if 'logged_in' in st.session_state and st.session_state['logged_in'] == 'admin':
            admin_page()
        else:
            st.error("Admin access only. Please log in as admin.")
    elif choice == "User":
        if 'logged_in' in st.session_state:
            user_page()
        else:
            st.error("Please log in to access the user page.")

# Run the app
if __name__ == '__main__':
    main()
