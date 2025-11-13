'''import streamlit as st
import pandas as pd
from recommender import create_recommender

@st.cache_data
def load_data():
    data_movies = pd.read_csv("data/movies.csv")
    data_ratings = pd.read_csv("data/ratings.csv.zip")
    merged_data = pd.merge(data_ratings, data_movies, on='movieId')
    merged_data['genres'] = merged_data['genres'].apply(lambda x: ' '.join(sorted(x.split('|'))))
    return merged_data

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommender System")
st.markdown("Get personalized movie recommendations using a Neural Collaborative Filtering model!")

# Load and initialize
data = load_data()
recommender = create_recommender(data)

# Input
user_id = st.number_input("🔢 Enter User ID", min_value=1, max_value=int(data['userId'].max()), step=1)

# Button
if st.button("🎯 Get Recommendations"):
    with st.spinner("Fetching recommendations..."):
        recs = recommender.get_recommendations(user_id=int(user_id), n_recommendations=10)
        if recs:
            st.success("Here are your recommendations!")
            for r in recs:
                st.subheader(r['title'])
                st.write(f"⭐ **Predicted Rating:** {r['predicted_rating']:.2f}")
                st.write(f"🎞 **Genres:** {r['genres']}")
                st.markdown("---")
        else:
            st.warning("No recommendations found for this user.")
'''


# app.py
import streamlit as st
import pandas as pd
from recommender import MovieRecommender

# Load data (cache enabled)
@st.cache_data
def load_data():
    ratings = pd.read_csv("/home/parth/Downloads/ml-latest-small/ratings.csv")
    movies = pd.read_csv("/home/parth/Downloads/ml-latest-small/movies.csv")
    links = pd.read_csv("/home/parth/Downloads/ml-latest-small/links.csv")
    tags = pd.read_csv("/home/parth/Downloads/ml-latest-small/tags.csv")

    links['tmdbId'].fillna(-1, inplace=True)
    merged = pd.merge(ratings, movies, on='movieId')
    merged['genres'] = merged['genres'].apply(lambda x: ' '.join(sorted(x.split('|'))))
    return merged, ratings, movies

# Page setup
st.set_page_config(page_title="\U0001F3AC Movie Recommender", layout="wide")
st.title("\U0001F3AC Personalized Movie Recommender (NCF-Based)")
st.markdown("Get personalized movie recommendations using a Neural Collaborative Filtering model!")

# Load & initialize
merged_data, ratings_df, movies_df = load_data()
recommender = MovieRecommender(merged_data)

# Sidebar - User Input
st.sidebar.header("\U0001F527 Settings")
user_id = st.sidebar.number_input("\U0001F464 Enter User ID", min_value=1, max_value=int(merged_data['userId'].max()), step=1)
n_recs = st.sidebar.slider("\U0001F3AF Number of Recommendations", min_value=5, max_value=20, value=10)

# Main area
if st.sidebar.button("\u2728 Recommend"):
    with st.spinner("Generating recommendations..."):
        recommendations = recommender.get_recommendations(user_id=user_id, n_recommendations=n_recs)

        if recommendations:
            st.success(f"Top {n_recs} Movie Recommendations for User ID: {user_id}")
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"{i}. {rec['title']}"):
                    st.markdown(f"**\U00002B50 Predicted Rating:** {rec['predicted_rating']:.2f}")
                    st.markdown(f"**\U0001F39E Genres:** {rec['genres']}")
        else:
            st.warning("\U0001F6AB No recommendations found. Please try another User ID.")

# Optional: show actual rated movies
if st.checkbox("\U0001F4C4 Show Movies Rated by This User"):
    if user_id in recommender.user_mapping:
        actual_user_id = user_id
        rated_movies = merged_data[merged_data['userId'] == actual_user_id][['title', 'rating', 'genres']]
        st.write(f"Showing {len(rated_movies)} movies rated by User ID {user_id}:")
        st.dataframe(rated_movies.sort_values(by='rating', ascending=False).reset_index(drop=True))
    else:
        st.warning("User not found in the dataset.")
