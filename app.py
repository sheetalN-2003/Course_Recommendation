import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from streamlit_chat import message
import nltk

# Download NLTK data (if not already downloaded)
nltk.download()

# Initialize stemmer for text preprocessing
stemmer = PorterStemmer()

# Preprocessing function to clean and normalize text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    words = word_tokenize(text)  # Tokenize the text into words
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Remove stop words and apply stemming
    return " ".join(words)  # Join words back into a string

# Load and preprocess the course data
def load_data():
    # Check if course data is already stored in session state
    if 'course_data' not in st.session_state:
        st.session_state['course_data'] = pd.DataFrame(columns=[
            'Course Name', 'Difficulty Level', 'Course Description', 'Skills', 'Ratings', 'Course URL'
        ])

    # Upload the CSV file with course data
    uploaded_file = st.file_uploader("Upload your course dataset (CSV format)", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)  # Read the CSV file
            st.write("Columns in the uploaded file:", df.columns)

            # Clean up column names and ensure required columns are present
            df.columns = df.columns.str.strip()  # Remove extra spaces from column names
            required_columns = ['Course Name', 'Difficulty Level', 'Course Description', 'Skills', 'Ratings', 'Course URL']
            missing_columns = [col for col in required_columns if col not in df.columns]

            # Handle missing columns
            if missing_columns:
                st.warning(f"The uploaded dataset is missing the following columns: {', '.join(missing_columns)}")
                for col in missing_columns:
                    if col == 'Ratings':
                        df['Ratings'] = "N/A"  # Add default value for missing Ratings
                    elif col == 'Course URL':
                        df['Course URL'] = "#"  # Add default URL for missing Course URL
                    elif col in ['Course Name', 'Difficulty Level', 'Course Description', 'Skills']:
                        df[col] = ""  # Add empty values for missing text fields

            df.fillna('', inplace=True)  # Fill missing values with empty strings
            # Create a new 'tags' column by combining relevant columns for better matching
            df['tags'] = (df['Course Name'].astype(str) + ' ' +
                           df['Difficulty Level'].astype(str) + ' ' +
                           df['Course Description'].astype(str) + ' ' +
                           df['Skills'].astype(str))
            df['tags'] = df['tags'].apply(preprocess_text)  # Apply text preprocessing
            st.session_state['course_data'] = df  # Save the processed data to session state
        except Exception as e:
            st.error(f"Error reading the file: {e}")
    return st.session_state['course_data']

# Recommendation function using cosine similarity and TF-IDF
def recommend_courses(df, vectorizer, query, num_recommendations=5):
    # Preprocess the user's query
    query_processed = preprocess_text(query)

    # Vectorize the dataset and the user's query
    vectors = vectorizer.fit_transform(df['tags']).toarray()  # Vectorize the tags column of the dataset
    query_vector = vectorizer.transform([query_processed]).toarray()  # Vectorize the user's query

    # Compute cosine similarity between the query and all courses
    similarities = cosine_similarity(query_vector, vectors).flatten()

    # Get the top N similar courses based on cosine similarity
    similar_courses = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:num_recommendations]

    # Create a list of recommendations
    recommendations = []
    for i, score in similar_courses:
        recommendations.append({
            'Course Name': df.iloc[i]['Course Name'],
            'Rating': df.iloc[i]['Ratings'],
            'URL': df.iloc[i]['Course URL'],
            'Similarity Score': round(score, 2)
        })
    return recommendations

# Search bar function to allow users to search for courses based on keywords
def search_bar(df):
    st.subheader("Search Courses")
    search_query = st.text_input("Enter a keyword or skill to search for courses:")
    if st.button("Search"):
        if search_query:
            # Filter courses based on the search query
            results = df[df['tags'].str.contains(preprocess_text(search_query), case=False)]
            if not results.empty:
                st.write("### Search Results")
                for _, row in results.iterrows():
                    st.markdown(f"- [{row['Course Name']}]({row['Course URL']}) (Rating: {row['Ratings']})")
            else:
                st.write("No courses found for the given query.")
        else:
            st.error("Please enter a keyword or skill to search.")  # Error if no input is given

# Chatbot interface for recommending courses based on user input
def chatbot_interface(df):
    st.markdown("<h2 style='text-align: center;'>Course Recommendation Chatbot</h2>", unsafe_allow_html=True)

    # Initialize messages in session state if not already done
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous conversation messages (if any)
    for msg in st.session_state.messages:
        message(msg['content'], is_user=msg['is_user'])

    # Input field for user to enter their message
    user_input = st.text_input("You:", key="chat_input", placeholder="Ask about courses or skills...")
    if st.button("Send", key="send_button") and user_input:
        st.session_state.messages.append({"content": user_input, "is_user": True})

        # Initialize TfidfVectorizer for text vectorization
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

        # Call the recommendation function
        recommendations = recommend_courses(df, vectorizer, user_input)

        # Create a response based on the recommendations
        if recommendations:
            bot_response = "Here are some course recommendations for you:\n"
            for rec in recommendations:
                bot_response += f"- [{rec['Course Name']}]({rec['URL']}) (Rating: {rec['Rating']}, Similarity: {rec['Similarity Score']})\n"
        else:
            bot_response = "Sorry, I couldn't find any matching courses."

        # Append the bot's response to the session state
        st.session_state.messages.append({"content": bot_response, "is_user": False})
        message(bot_response, is_user=False)

# Simulate course enrollment
def enroll_in_course(course_name):
    if 'enrolled_courses' not in st.session_state:
        st.session_state.enrolled_courses = []
    if course_name not in st.session_state.enrolled_courses:
        st.session_state.enrolled_courses.append(course_name)
        st.success(f"You have successfully enrolled in {course_name}!")
    else:
        st.warning(f"You are already enrolled in {course_name}.")

# Display user dashboard with enrolled courses and progress
def user_dashboard():
    st.subheader("Your Dashboard")
    if 'enrolled_courses' in st.session_state and st.session_state.enrolled_courses:
        st.write("### Enrolled Courses")
        for course in st.session_state.enrolled_courses:
            st.write(f"- {course}")
        st.write("### Progress Tracker")
        st.progress(50)  # Simulate 50% progress
    else:
        st.write("You are not enrolled in any courses yet.")

# Display popular courses based on ratings
def display_popular_courses(df):
    st.subheader("Popular Courses")
    popular_courses = df.sort_values(by='Ratings', ascending=False).head(5)
    for _, row in popular_courses.iterrows():
        st.markdown(f"- [{row['Course Name']}]({row['Course URL']}) (Rating: {row['Ratings']})")

# Main app function to set up the Streamlit UI and navigation
def main():
    st.set_page_config(page_title="Course Recommendation System", layout="wide")

    # Initialize session state variables for user management
    if 'users' not in st.session_state:
        st.session_state['users'] = []
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = None
    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False

    menu = ["Home", "Register", "Login", "Admin", "User"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home page content
    if choice == "Home":
        st.title("Welcome to the Course Recommendation System")
        st.write("This platform helps you discover courses tailored to your interests.")

    # Register page for new users
    elif choice == "Register":
        st.subheader("Register")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Register"):
            if password == confirm_password:
                users = st.session_state.get('users', [])
                if any(user['username'] == username for user in users):
                    st.error("Username already exists. Please choose a different username.")
                else:
                    users.append({"username": username, "password": password})
                    st.session_state['users'] = users
                    st.success("Registration successful!")
            else:
                st.error("Passwords do not match.")  # Error if passwords don't match

    # Login page for existing users
    elif choice == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            users = st.session_state.get('users', [])
            if any(user['username'] == username and user['password'] == password for user in users):
                st.session_state['logged_in'] = username
                st.success(f"Welcome, {username}!")
            else:
                st.error("Invalid username or password.")  # Error for invalid login credentials

    # Admin page for managing users (restricted access)
    elif choice == "Admin":
        if st.session_state.get('admin_logged_in'):
            st.subheader("Admin Page")
            users = st.session_state.get('users', [])
            if users:
                st.write("### Registered Users")
                for user in users:
                    st.write(f"- {user['username']}")
            else:
                st.write("No registered users.")  # Message if there are no registered users
        else:
            st.error("You do not have permission to access this page.")  # Restrict access

    # User page for logged-in users to explore courses
    elif choice == "User":
        if st.session_state['logged_in']:
            st.subheader("User Page")
            st.write("Welcome to the user page. Use the navigation options to explore courses.")
            df = load_data()
            if not df.empty:
                st.write("### Explore Courses")
                search_bar(df)  # Display the search bar for course search
                chatbot_interface(df)  # Display the chatbot interface for course recommendations
                display_popular_courses(df)  # Display popular courses
                user_dashboard()  # Display user dashboard
        else:
            st.error("Please log in to access the user page.")  # Error if user is not logged in

# Run the Streamlit app
if __name__ == '__main__':
    main()
