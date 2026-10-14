import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


                                                              
                  
                                                              

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"


def find_csv_file(filename):
    """Find expected filenames and common uploaded/downloaded filename variants."""
    aliases = {
        "clean_movies.csv": ["clean_movies.csv", "clean_movies(1).csv", "movies.csv"],
        "clean_ratings.csv": ["clean_ratings.csv", "clean_ratings(1).csv", "ratings.csv"],
        "evaluation_results.csv": ["evaluation_results.csv", "evaluation_results(1).csv"],
    }
    candidates = aliases.get(filename, [filename])
    for candidate in candidates:
        expected_path = RESULTS_DIR / candidate
        if expected_path.is_file():
            return expected_path
        direct_path = PROJECT_DIR / candidate
        if direct_path.is_file():
            return direct_path

    matches = sorted(
        path for candidate in candidates
        for path in PROJECT_DIR.rglob(candidate)
        if path.is_file()
    )
    # Remove duplicate paths if a candidate was found by more than one route.
    matches = list(dict.fromkeys(matches))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename}. Put the CSV in {RESULTS_DIR} "
            f"or in the same folder as app.py."
        )

    if len(matches) > 1:
        match_list = "\n".join(str(path) for path in matches)
        raise FileNotFoundError(
            f"Found multiple files named {filename}. Keep one copy in "
            f"{RESULTS_DIR} or update the app to select the correct file:\n"
            f"{match_list}"
        )

    return matches[0]


                                                              
              
                                                              

@st.cache_data
def load_data():
    ratings = pd.read_csv(find_csv_file("clean_ratings.csv"))
    movies = pd.read_csv(find_csv_file("clean_movies.csv"))

    return ratings, movies


ratings, movies = load_data()


                                                              
                              
                                                              

@st.cache_resource
def create_content_model(movies):
    movies = movies.copy()

                                                  
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

                              
    content_results["Content Score"] = (
        content_results["Score"] /
        content_results["Score"].max()
    )

                                    
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


                                                              
                   
                                                              

st.set_page_config(
    page_title="CINEMATCH | Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Netflix-inspired dark interface. All recommendation logic above remains unchanged.
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
:root { --red:#e50914; --bg:#080808; --panel:#171717; --muted:#b3b3b3; }
html, body, [class*="css"] { font-family: Inter, Arial, sans-serif; }
.stApp { background: #080808; color: #f5f5f1; }
[data-testid="stHeader"] { background: rgba(8,8,8,.96); }
[data-testid="stSidebar"] { background: #101010; border-right: 1px solid #262626; }
[data-testid="stSidebar"] * { color: #f5f5f1; }
.block-container { padding-top: 1.2rem; max-width: 1450px; }
.brand { color: #e50914; font-size: 1.55rem; font-weight: 900; letter-spacing: 2px; }
.hero { background: linear-gradient(90deg,rgba(0,0,0,.96) 0%,rgba(0,0,0,.78) 48%,rgba(0,0,0,.25) 100%), radial-gradient(circle at 78% 20%,#632126 0,#251014 28%,#101010 66%); border-radius: 12px; padding: 42px 38px; border: 1px solid #292929; min-height: 235px; }
.hero h1 { font-size: clamp(2rem,4vw,3.5rem); line-height: 1.05; font-weight: 900; margin: 6px 0 14px; }
.hero p { color: #dedede; max-width: 680px; font-size: 1.02rem; }
.eyebrow { color: #ff5b63; font-size: .76rem; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; }
.section-title { font-size: 1.35rem; font-weight: 800; margin: 22px 0 12px; }
.metric-card { background: linear-gradient(145deg,#202020,#121212); border: 1px solid #303030; border-radius: 10px; padding: 17px 19px; min-height: 96px; }
.metric-label { color: #b3b3b3; font-size: .82rem; }
.metric-value { color: white; font-size: 1.8rem; font-weight: 800; margin-top: 4px; }
.movie-card { background: linear-gradient(145deg,#262626,#111); border: 1px solid #303030; border-radius: 9px; padding: 15px; min-height: 150px; transition: transform .15s ease; }
.movie-card:hover { border-color: #e50914; }
.movie-number { color: #e50914; font-weight: 900; font-size: .8rem; letter-spacing: 1px; }
.movie-name { color: #fff; font-size: 1rem; font-weight: 800; margin: 8px 0; line-height: 1.25; }
.movie-genre { color: #bdbdbd; font-size: .78rem; min-height: 32px; }
.score { display:inline-block; background:#e50914; color:white; padding:4px 8px; border-radius:4px; font-size:.75rem; font-weight:800; margin-top:10px; }
div.stButton > button { background:#e50914; color:white; border:0; border-radius:5px; font-weight:800; min-height:44px; }
div.stButton > button:hover { background:#b80710; color:white; border:0; }
.stSelectbox label, .stRadio label, .stSlider label { color:#dedede !important; }
[data-testid="stMetric"] { background:#171717; padding:12px; border:1px solid #303030; border-radius:8px; }
[data-testid="stMetricLabel"] { color:#b3b3b3 !important; }
[data-testid="stMetricValue"] { color:#fff !important; }
hr { border-color:#303030; }
.small-note { color:#999; font-size:.78rem; }
</style>
''', unsafe_allow_html=True)

# Sidebar: controls are kept simple and familiar.
with st.sidebar:
    st.markdown('<div class="brand">CINEMATCH</div>', unsafe_allow_html=True)
    st.caption("YOUR NEXT FAVOURITE FILM STARTS HERE")
    st.divider()
    st.markdown("### 🎛️ Recommendation settings")
    selected_user = st.selectbox("Choose a user profile", sorted(ratings["user_id"].unique()))
    movie_titles = sorted(movies["title"].dropna().unique())
    selected_movie = st.selectbox("A movie you like", movie_titles)
    recommendation_type = st.radio(
        "Recommendation engine",
        ["Hybrid", "Content-Based", "Collaborative Filtering"],
        help="Hybrid combines movie genre similarity with preferences from similar users.",
    )
    number_of_recommendations = st.slider("How many movies?", 5, 15, 10)
    generate = st.button("▶  FIND MY MOVIES", use_container_width=True)
    st.divider()
    st.markdown("**About the engine**")
    st.caption("Content-based · Collaborative · Hybrid")
    st.caption("Built with Python, pandas, scikit-learn and Streamlit.")

st.markdown('<div class="brand">CINEMATCH</div>', unsafe_allow_html=True)
st.markdown(f'''<div class="hero">
<div class="eyebrow">A personalised movie experience</div>
<h1>Movies picked<br>for <span style="color:#e50914">your taste.</span></h1>
<p>Discover your next watch with recommendations powered by movie genres and the tastes of viewers like you. Choose a film and profile in the left panel to get started.</p>
</div>''', unsafe_allow_html=True)

st.markdown('<div class="section-title">Your movie universe</div>', unsafe_allow_html=True)
metric_cols = st.columns(3)
metrics = [("MOVIES IN CATALOGUE", f"{len(movies):,}"), ("VIEWER PROFILES", f"{ratings['user_id'].nunique():,}"), ("RATINGS ANALYSED", f"{len(ratings):,}")]
for col, (label, value) in zip(metric_cols, metrics):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">🔥 Popular picks in the catalogue</div>', unsafe_allow_html=True)
# Display the most-rated titles as a browse row; these are catalogue popularity, not personalised predictions.
popular_ids = ratings.groupby("movie_id").size().sort_values(ascending=False).head(5)
popular = movies[movies["movie_id"].isin(popular_ids.index)].copy()
popular["rating_count"] = popular["movie_id"].map(popular_ids)
popular = popular.sort_values("rating_count", ascending=False)
pop_cols = st.columns(5)
for i, (_, film) in enumerate(popular.iterrows()):
    with pop_cols[i % 5]:
        genre = str(film.get("genres", ""))
        st.markdown(f'''<div class="movie-card"><div class="movie-number">TOP PICK 0{i+1}</div><div class="movie-name">{film['title']}</div><div class="movie-genre">{genre}</div><div class="score">{int(film['rating_count']):,} ratings</div></div>''', unsafe_allow_html=True)

st.markdown('<div class="section-title">🎯 Made for your next movie night</div>', unsafe_allow_html=True)
st.caption("Select your profile, choose a movie you enjoy, and select the recommendation method from the sidebar.")
if generate:
    with st.spinner("Finding movies that match your taste..."):
        if recommendation_type == "Content-Based":
            results = content_recommendations(selected_movie, number_of_recommendations)
        elif recommendation_type == "Collaborative Filtering":
            results = collaborative_recommendations(selected_user, number_of_recommendations)
        else:
            results = hybrid_recommendations(selected_user, selected_movie, number_of_recommendations)

    if results.empty:
        st.warning("No recommendations were found for this selection. Try another movie or user profile.")
    else:
        st.success(f"{len(results)} recommendations ready · {recommendation_type}")
        display = results.copy()
        # Render as visual movie tiles rather than a plain dataframe.
        card_cols = st.columns(5)
        for idx, (_, film) in enumerate(display.iterrows()):
            col = card_cols[idx % 5]
            with col:
                title = str(film.get("Movie", "Untitled"))
                genre = str(film.get("Genre", "Genre not available"))
                if "Hybrid Score" in film.index:
                    score_label = f"Match {float(film['Hybrid Score']):.0%}"
                elif "Score" in film.index:
                    score_label = f"Score {float(film['Score']):.3f}"
                else:
                    score_label = "Recommended"
                st.markdown(f'''<div class="movie-card"><div class="movie-number">CINEMATCH PICK {idx+1:02d}</div><div class="movie-name">{title}</div><div class="movie-genre">{genre}</div><div class="score">{score_label}</div></div>''', unsafe_allow_html=True)
        with st.expander("See recommendation scores and details"):
            st.dataframe(results, use_container_width=True, hide_index=True)
else:
    st.info("👈 Set your preferences in the sidebar, then press **FIND MY MOVIES** to generate personalised recommendations.")

with st.expander("📊 Model evaluation"):
    st.write("The evaluation file reports Precision@10, Recall@10, F1-score@10 and Mean Squared Error (MSE).")
    try:
        evaluation_file = find_csv_file("evaluation_results.csv")
        evaluation = pd.read_csv(evaluation_file)
        if {"Metric", "Value"}.issubset(evaluation.columns):
            eval_cols = st.columns(min(4, len(evaluation)))
            for i, (_, row) in enumerate(evaluation.iterrows()):
                with eval_cols[i % len(eval_cols)]:
                    val = float(row["Value"])
                    formatted = f"{val:.4f}"
                    st.markdown(f'<div class="metric-card"><div class="metric-label">{row["Metric"]}</div><div class="metric-value" style="font-size:1.45rem">{formatted}</div></div>', unsafe_allow_html=True)
        else:
            st.dataframe(evaluation, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.info("Evaluation results file was not found. Add evaluation_results.csv to the project folder to display metrics.")

st.divider()
st.markdown('<div class="small-note">CINEMATCH · BICT242 Data Scalability and Analytics · A student movie recommendation project. Inspired by streaming-service interfaces; not affiliated with Netflix.</div>', unsafe_allow_html=True)
