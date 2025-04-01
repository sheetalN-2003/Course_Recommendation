# THIS MUST BE THE VERY FIRST LINE IN YOUR SCRIPT
import streamlit as st

# Set page config IMMEDIATELY after streamlit import
st.set_page_config(
    page_title="AI Learning Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import all other dependencies
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.users = {
        'admin': {
            'password': 'admin123', 
            'role': 'admin', 
            'progress': {}, 
            'preferences': {},
            'joined_date': datetime.now().strftime("%Y-%m-%d")
        }
    }
    st.session_state.logged_in = None
    st.session_state.courses = pd.DataFrame(columns=[
        'Course Name', 'Description', 'Category', 'Skills', 'Difficulty Level',
        'Duration', 'Price', 'Instructor', 'Ratings', 'Course URL', 'Created At'
    ])
    st.session_state.messages = []
    st.session_state.learning_paths = {}

# Data validation function
def validate_course_data(df):
    required_columns = {
        'Course Name': str,
        'Description': str,
        'Category': str,
        'Skills': (list, str),
        'Difficulty Level': str,
        'Course URL': str
    }
    
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return False
    
    return True

# Data preprocessing function
def preprocess_course_data(df):
    # Ensure Skills column is properly formatted
    if 'Skills' in df.columns:
        df['Skills'] = df['Skills'].apply(
            lambda x: [x.strip() for x in x.split(',')] if isinstance(x, str) else x if isinstance(x, list) else []
        )
    
    # Set default values for optional columns
    if 'Ratings' not in df.columns:
        df['Ratings'] = 4.0
    else:
        df['Ratings'] = pd.to_numeric(df['Ratings'], errors='coerce').fillna(4.0)
    
    if 'Price' not in df.columns:
        df['Price'] = 0.0
    
    return df

# Authentication functions
def register():
    st.subheader("Register New User")
    with st.form("register_form"):
        new_username = st.text_input("Enter Username")
        new_password = st.text_input("Enter Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")
        
        if submitted:
            if new_password != confirm_password:
                st.error("Passwords don't match!")
            elif new_username in st.session_state.users:
                st.warning("Username already exists!")
            else:
                st.session_state.users[new_username] = {
                    'password': new_password,
                    'role': 'user',
                    'progress': {},
                    'preferences': {},
                    'joined_date': datetime.now().strftime("%Y-%m-%d")
                }
                st.success("Registration successful! You can now log in.")
                time.sleep(1)
                st.rerun()

def login():
    st.subheader("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if (username in st.session_state.users and 
                st.session_state.users[username]['password'] == password):
                st.session_state.logged_in = username
                st.success(f"Welcome back, {username}!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid credentials!")

def logout():
    st.session_state.logged_in = None
    st.success("Logged out successfully.")
    time.sleep(1)
    st.rerun()

# Admin course management functions
def admin_add_course():
    st.subheader("Add New Course")
    with st.form("add_course_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Course Name*")
            category = st.selectbox("Category*", ["Programming", "Data Science", "Business", "Design", "Other"])
            skills = st.text_input("Skills (comma separated)*", "Python, Data Analysis")
            difficulty = st.selectbox("Difficulty Level*", ["Beginner", "Intermediate", "Advanced"])
            url = st.text_input("Course URL*", "https://example.com/course")
        
        with col2:
            description = st.text_area("Description*")
            duration = st.number_input("Duration (hours)", min_value=1, value=10)
            price = st.number_input("Price ($)", min_value=0, value=0)
            instructor = st.text_input("Instructor")
            rating = st.slider("Rating", 1.0, 5.0, 4.5, 0.1)
        
        if st.form_submit_button("Add Course"):
            if not all([name, category, skills, difficulty, url, description]):
                st.error("Please fill all required fields (marked with *)")
            else:
                new_course = {
                    'Course Name': name,
                    'Description': description,
                    'Category': category,
                    'Skills': skills,
                    'Difficulty Level': difficulty,
                    'Duration': duration,
                    'Price': price,
                    'Instructor': instructor if instructor else "Not specified",
                    'Ratings': rating,
                    'Course URL': url,
                    'Created At': datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                
                # Add to courses DataFrame
                st.session_state.courses = pd.concat([
                    st.session_state.courses,
                    pd.DataFrame([new_course])
                ], ignore_index=True)
                
                # Preprocess the data
                st.session_state.courses = preprocess_course_data(st.session_state.courses)
                
                st.success("Course added successfully!")
                time.sleep(1)
                st.rerun()

def admin_manage_courses():
    st.subheader("Manage Courses")
    
    if not st.session_state.courses.empty:
        st.dataframe(st.session_state.courses)
        
        # Course deletion
        with st.expander("Delete Courses"):
            courses_to_delete = st.multiselect(
                "Select courses to delete",
                options=st.session_state.courses['Course Name'].tolist()
            )
            
            if st.button("Delete Selected Courses"):
                st.session_state.courses = st.session_state.courses[
                    ~st.session_state.courses['Course Name'].isin(courses_to_delete)
                ]
                st.success(f"Deleted {len(courses_to_delete)} courses")
                time.sleep(1)
                st.rerun()
    else:
        st.warning("No courses available. Add some courses first.")

# Course recommendation function
def recommend_courses(df, query, num_recommendations=5):
    if df.empty:
        return []
    
    # Create tags from course information
    df['tags'] = df[['Course Name', 'Category', 'Skills', 'Difficulty Level']].apply(
        lambda x: ' '.join(x.astype(str)), axis=1
    )
    
    # Simple recommendation based on category/skill match
    recommendations = df[
        df['Category'].str.contains(query, case=False) | 
        df['Skills'].apply(lambda x: query.lower() in [s.lower() for s in x])
    ].head(num_recommendations)
    
    return recommendations.to_dict('records')

# Course marketplace for users
def course_marketplace():
    st.subheader("Course Marketplace")
    
    if st.session_state.courses.empty:
        st.warning("No courses available yet. Please check back later.")
        return
    
    # Search and filters
    search_query = st.text_input("Search courses")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.selectbox(
            "Category",
            ["All"] + list(st.session_state.courses['Category'].unique())
        )
    with col2:
        difficulty = st.selectbox(
            "Difficulty Level",
            ["All"] + list(st.session_state.courses['Difficulty Level'].unique())
        )
    with col3:
        min_rating = st.slider("Minimum Rating", 1.0, 5.0, 3.0, 0.1)
    
    # Apply filters
    filtered_courses = st.session_state.courses.copy()
    
    if search_query:
        filtered_courses = filtered_courses[
            filtered_courses['Course Name'].str.contains(search_query, case=False) |
            filtered_courses['Description'].str.contains(search_query, case=False) |
            filtered_courses['Skills'].apply(lambda x: any(search_query.lower() in s.lower() for s in x))
        ]
    
    if category != "All":
        filtered_courses = filtered_courses[filtered_courses['Category'] == category]
    
    if difficulty != "All":
        filtered_courses = filtered_courses[filtered_courses['Difficulty Level'] == difficulty]
    
    filtered_courses = filtered_courses[filtered_courses['Ratings'] >= min_rating]
    
    # Display courses
    if not filtered_courses.empty:
        st.write(f"Found {len(filtered_courses)} courses:")
        
        for _, row in filtered_courses.iterrows():
            with st.expander(f"{row['Course Name']} - ⭐ {row['Ratings']}"):
                st.write(f"**Category:** {row['Category']}")
                st.write(f"**Difficulty:** {row['Difficulty Level']}")
                st.write(f"**Skills:** {', '.join(row['Skills'])}")
                st.write(f"**Duration:** {row['Duration']} hours")
                st.write(f"**Price:** ${row['Price']}")
                st.write(f"**Instructor:** {row['Instructor']}")
                st.write(f"**Description:** {row['Description']}")
                
                if pd.notna(row['Course URL']):
                    st.write(f"[Course Link]({row['Course URL']})")
                
                if st.button("Enroll in Course", key=f"enroll_{row['Course Name']}"):
                    if st.session_state.logged_in and st.session_state.logged_in != "guest":
                        user = st.session_state.users[st.session_state.logged_in]
                        if 'progress' not in user:
                            user['progress'] = {}
                        user['progress'][row['Course Name']] = 0
                        st.success("Successfully enrolled in course!")
                    else:
                        st.warning("Please log in to enroll in courses")
    else:
        st.write("No courses match your search. Try different filters.")

# Learning path creation
def create_learning_path():
    st.subheader("Create Learning Path")
    
    if st.session_state.courses.empty:
        st.warning("No courses available to create learning paths.")
        return
    
    with st.form("learning_path_form"):
        path_name = st.text_input("Learning Path Name")
        goal = st.text_area("Learning Goal")
        selected_courses = st.multiselect(
            "Select Courses",
            options=st.session_state.courses['Course Name'].tolist()
        )
        
        if st.form_submit_button("Create Learning Path"):
            if not path_name or not goal or not selected_courses:
                st.error("Please fill all fields and select at least one course")
            else:
                path_id = f"path_{int(time.time())}"
                st.session_state.learning_paths[path_id] = {
                    'name': path_name,
                    'goal': goal,
                    'courses': selected_courses,
                    'progress': 0,
                    'created_at': datetime.now().strftime("%Y-%m-%d")
                }
                st.success("Learning path created successfully!")
                st.rerun()

# User progress tracking
def user_progress():
    st.subheader("My Learning Progress")
    
    if st.session_state.logged_in and st.session_state.logged_in != "guest":
        user = st.session_state.users[st.session_state.logged_in]
        
        if 'progress' in user and user['progress']:
            st.write("### Your Enrolled Courses")
            for course, progress in user['progress'].items():
                st.write(f"- **{course}**: {progress}% complete")
                st.progress(progress)
        else:
            st.warning("You haven't enrolled in any courses yet.")
        
        if st.session_state.learning_paths:
            st.write("### Your Learning Paths")
            for path_id, path in st.session_state.learning_paths.items():
                with st.expander(f"{path['name']} - {path['progress']}% complete"):
                    st.write(f"**Goal:** {path['goal']}")
                    st.write("**Courses:**")
                    for course in path['courses']:
                        st.write(f"- {course}")
                    if st.button("Mark as Completed", key=f"complete_{path_id}"):
                        path['progress'] = 100
                        st.success("Learning path completed!")
                        st.rerun()
    else:
        st.warning("Please log in to view your progress")

# Main application
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    if not st.session_state.logged_in:
        menu = ["Home", "Login", "Register"]
    elif st.session_state.users[st.session_state.logged_in]['role'] == "admin":
        menu = ["Dashboard", "Add Courses", "Manage Courses", "Logout"]
    else:
        menu = ["Dashboard", "Browse Courses", "Learning Paths", "My Progress", "Logout"]
    
    choice = st.sidebar.selectbox("Menu", menu)
    
    # Main content area
    if choice == "Home":
        st.title("AI Learning Platform")
        st.write("Welcome to our intelligent course recommendation system!")
        
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
                st.write(f"Total courses: {len(st.session_state.courses)}")
                st.write(f"Total users: {len(st.session_state.users)}")
            else:
                st.write("### Recommended Courses")
                if not st.session_state.courses.empty:
                    recs = recommend_courses(st.session_state.courses, "Python", 3)
                    for rec in recs:
                        st.write(f"- **{rec['Course Name']}** (Rating: {rec['Ratings']})")
    
    elif choice == "Add Courses" and st.session_state.logged_in == "admin":
        admin_add_course()
    
    elif choice == "Manage Courses" and st.session_state.logged_in == "admin":
        admin_manage_courses()
    
    elif choice == "Browse Courses":
        course_marketplace()
    
    elif choice == "Learning Paths" and st.session_state.logged_in and st.session_state.logged_in != "guest":
        tab1, tab2 = st.tabs(["My Paths", "Create New"])
        
        with tab1:
            if st.session_state.learning_paths:
                for path_id, path in st.session_state.learning_paths.items():
                    with st.expander(f"{path['name']} - {path['progress']}% complete"):
                        st.write(f"**Goal:** {path['goal']}")
                        st.write("**Courses:**")
                        for course in path['courses']:
                            st.write(f"- {course}")
            else:
                st.write("You haven't created any learning paths yet.")
        
        with tab2:
            create_learning_path()
    
    elif choice == "My Progress" and st.session_state.logged_in and st.session_state.logged_in != "guest":
        user_progress()
    
    elif choice == "Logout" and st.session_state.logged_in:
        logout()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **AI Learning Platform** v2.0  
    © 2023 All Rights Reserved
    """)

if __name__ == "__main__":
    main()
