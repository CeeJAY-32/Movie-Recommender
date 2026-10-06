import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"


def find_csv_file(filename):
    """Find a CSV in the expected results folder or elsewhere in the project."""
    expected_path = RESULTS_DIR / filename
    if expected_path.is_file():
        return expected_path

    matches = sorted(
        path
        for path in PROJECT_DIR.rglob(filename)
        if path.is_file()
    )

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename}. Place it in {RESULTS_DIR} "
            f"or somewhere under {PROJECT_DIR}."
        )

    if len(matches) > 1:
        match_list = "\n".join(str(path) for path in matches)
        raise FileNotFoundError(
            f"Found multiple files named {filename}. Keep one copy in "
            f"{RESULTS_DIR} or update the app to select the correct file:\n"
            f"{match_list}"
        )

    return matches[0]


# ============================================================
# 2. LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    ratings = pd.read_csv(find_csv_file("clean_ratings.csv"))
    movies = pd.read_csv(find_csv_file("clean_movies.csv"))

    return ratings, movies


ratings, movies = load_data()


# ============================================================
# 3. CONTENT-BASED RECOMMENDER
# ============================================================

@st.cache_resource
def create_content_model(movies):
    movies = movies.copy()

    # Use genres for content-based recommendations
    movies["genres"] = movies["genres"].fillna("")

    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(movies["genres"])

    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return similarity_matrix


content_similarity = create_content_model(movies)


def content_recommendations(movie_title, number=10):

    movie_matches = movies[
        movies["title"].str.lower() == movie_title.lower()
    ]

    if movie_matches.empty:
        return pd.DataFrame()

    movie_index = movie_matches.index[0]

    similarity_scores = list(
        enumerate(content_similarity[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in similarity_scores[1:number + 1]:

        recommendations.append({
            "Movie": movies.iloc[index]["title"],
            "Genre": movies.iloc[index]["genres"],
            "Score": round(float(score), 3)
        })

    return pd.DataFrame(recommendations)


# ============================================================
# 4. COLLABORATIVE FILTERING
# ============================================================

@st.cache_data
def create_user_movie_matrix(ratings):

    matrix = ratings.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    ).fillna(0)

    return matrix


user_movie_matrix = create_user_movie_matrix(ratings)


@st.cache_resource
def create_user_similarity(matrix):

    similarity = cosine_similarity(matrix)

    return similarity


user_similarity = create_user_similarity(user_movie_matrix)


def collaborative_recommendations(user_id, number=10):

    if user_id not in user_movie_matrix.index:
        return pd.DataFrame()

    user_index = user_movie_matrix.index.get_loc(user_id)

    similarity_scores = list(
        enumerate(user_similarity[user_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similar_users = similarity_scores[1:11]

    user_ratings = user_movie_matrix.loc[user_id]

    already_rated = set(
        user_ratings[user_ratings > 0].index
    )

    movie_scores = {}

    for similar_user_index, similarity_score in similar_users:

        similar_user_id = user_movie_matrix.index[
            similar_user_index
        ]

        similar_user_ratings = user_movie_matrix.loc[
            similar_user_id
        ]

        for movie_id, rating in similar_user_ratings.items():

            if rating >= 4 and movie_id not in already_rated:

                if movie_id not in movie_scores:
                    movie_scores[movie_id] = 0

                movie_scores[movie_id] += (
                    similarity_score * rating
                )

    sorted_movies = sorted(
        movie_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for movie_id, score in sorted_movies[:number]:

        movie_info = movies[
            movies["movie_id"] == movie_id
        ]

        if not movie_info.empty:

            recommendations.append({
                "Movie": movie_info.iloc[0]["title"],
                "Genre": movie_info.iloc[0]["genres"],
                "Score": round(float(score), 3)
            })

    return pd.DataFrame(recommendations)


# ============================================================
# 5. HYBRID RECOMMENDER
# ============================================================

def hybrid_recommendations(
    user_id,
    movie_title,
    number=10
):

    content_results = content_recommendations(
        movie_title,
        number=50
    )

    if content_results.empty:
        return pd.DataFrame()

    collaborative_results = collaborative_recommendations(
        user_id,
        number=50
    )

    # Normalize content scores
    content_results["Content Score"] = (
        content_results["Score"] /
        content_results["Score"].max()
    )

    # Normalize collaborative scores
    if not collaborative_results.empty:

        collaborative_results["Collaborative Score"] = (
            collaborative_results["Score"] /
            collaborative_results["Score"].max()
        )

    else:

        collaborative_results["Collaborative Score"] = 0

    collaborative_dictionary = dict(
        zip(
            collaborative_results["Movie"],
            collaborative_results["Collaborative Score"]
        )
    )

    final_results = []

    for _, row in content_results.iterrows():

        movie = row["Movie"]

        content_score = row["Content Score"]

        collaborative_score = collaborative_dictionary.get(
            movie,
            0
        )

        hybrid_score = (
            0.5 * content_score +
            0.5 * collaborative_score
        )

        final_results.append({
            "Movie": movie,
            "Genre": row["Genre"],
            "Content Score": round(
                float(content_score), 3
            ),
            "Collaborative Score": round(
                float(collaborative_score), 3
            ),
            "Hybrid Score": round(
                float(hybrid_score), 3
            )
        })

    results = pd.DataFrame(final_results)

    results = results.sort_values(
        "Hybrid Score",
        ascending=False
    )

    return results.head(number)


# ============================================================
# 6. STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# 7. TITLE
# ============================================================

st.title("🎬 Movie Recommendation System")

st.write(
    "A personalized movie recommendation system "
    "using Content-Based Filtering, Collaborative "
    "Filtering, and a Hybrid Recommendation approach."
)


st.divider()


# ============================================================
# 8. PROJECT INFORMATION
# ============================================================

st.subheader("📊 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Movies",
        len(movies)
    )

with col2:
    st.metric(
        "Users",
        ratings["user_id"].nunique()
    )

with col3:
    st.metric(
        "Ratings",
        len(ratings)
    )


st.divider()


# ============================================================
# 9. USER AND MOVIE SELECTION
# ============================================================

st.subheader("🎯 Generate Recommendations")

col1, col2 = st.columns(2)


with col1:

    selected_user = st.selectbox(
        "Select a User",
        sorted(
            ratings["user_id"].unique()
        )
    )


with col2:

    movie_titles = sorted(
        movies["title"].dropna().unique()
    )

    selected_movie = st.selectbox(
        "Select a Movie",
        movie_titles
    )


# ============================================================
# 10. RECOMMENDATION TYPE
# ============================================================

recommendation_type = st.radio(
    "Choose Recommendation Method",
    [
        "Content-Based",
        "Collaborative Filtering",
        "Hybrid"
    ],
    horizontal=True
)


number_of_recommendations = st.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=15,
    value=10
)


# ============================================================
# 11. GENERATE BUTTON
# ============================================================

if st.button(
    "🚀 Generate Recommendations",
    type="primary"
):

    with st.spinner(
        "Generating recommendations..."
    ):

        if recommendation_type == "Content-Based":

            results = content_recommendations(
                selected_movie,
                number_of_recommendations
            )

            st.success(
                "Content-based recommendations generated!"
            )

            if not results.empty:

                st.dataframe(
                    results,
                    use_container_width=True
                )

        elif recommendation_type == "Collaborative Filtering":

            results = collaborative_recommendations(
                selected_user,
                number_of_recommendations
            )

            st.success(
                "Collaborative recommendations generated!"
            )

            if not results.empty:

                st.dataframe(
                    results,
                    use_container_width=True
                )

        else:

            results = hybrid_recommendations(
                selected_user,
                selected_movie,
                number_of_recommendations
            )

            st.success(
                "Hybrid recommendations generated!"
            )

            if not results.empty:

                st.dataframe(
                    results,
                    use_container_width=True
                )


# ============================================================
# 12. MODEL EVALUATION
# ============================================================

st.divider()

st.subheader("📈 Model Evaluation")

st.write(
    "The recommendation system was evaluated using "
    "Precision@10, Recall@10, F1-score@10 and MSE."
)

try:
    evaluation_file = find_csv_file("evaluation_results.csv")
except FileNotFoundError:
    evaluation_file = None

if evaluation_file is not None:

    evaluation = pd.read_csv(
        evaluation_file
    )

    st.dataframe(
        evaluation,
        use_container_width=True
    )

else:

    st.info(
        "Evaluation results file was not found."
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "BICT242 Data Scalability and Analytics — "
    "Movie Recommendation System"
)