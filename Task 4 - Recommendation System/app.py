"""
CineMatch — Hybrid Movie Recommendation System
================================================
Streamlit application combining Content-Based Filtering (TF-IDF + Cosine
Similarity) and Collaborative Filtering (SVD) into an explainable hybrid
recommendation engine.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from engine.data_manager import DataManager
from engine.content_based import ContentBasedRecommender
from engine.collaborative import CollaborativeRecommender
from engine.hybrid import HybridRecommender

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch • Smart Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main .block-container { padding-top: 1.5rem; max-width: 1200px; }

/* Header */
.hero-title {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0; line-height: 1.2;
}
.hero-sub {
    font-size: 1.05rem; color: #8892b0; margin-top: 0.25rem;
    margin-bottom: 1.5rem;
}

/* Movie cards */
.movie-card {
    background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(102,126,234,0.15);
    border-radius: 16px; padding: 1.4rem; margin-bottom: 1rem;
    transition: all 0.3s ease; position: relative; overflow: hidden;
}
.movie-card:hover {
    border-color: rgba(102,126,234,0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(102,126,234,0.15);
}
.movie-card::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 4px; height: 100%; border-radius: 4px 0 0 4px;
    background: linear-gradient(180deg, #667eea, #764ba2);
}
.movie-title { font-size: 1.15rem; font-weight: 700; color: #e6f1ff; margin-bottom: 0.3rem; }
.movie-year { font-size: 0.8rem; color: #8892b0; font-weight: 500; }
.movie-overview { font-size: 0.82rem; color: #a8b2d1; line-height: 1.5; margin: 0.6rem 0; }
.genre-pill {
    display: inline-block; padding: 3px 10px; margin: 2px 3px 2px 0;
    border-radius: 20px; font-size: 0.7rem; font-weight: 600;
    background: rgba(102,126,234,0.15); color: #667eea;
    border: 1px solid rgba(102,126,234,0.25);
}
.score-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2); color: #fff;
}
.reason-text {
    font-size: 0.78rem; color: #64ffda; font-style: italic;
    margin-top: 0.5rem; padding: 0.5rem 0.7rem;
    background: rgba(100,255,218,0.05); border-radius: 8px;
    border-left: 3px solid #64ffda;
}

/* Metric boxes */
.metric-box {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid rgba(102,126,234,0.2); border-radius: 14px;
    padding: 1.2rem; text-align: center;
}
.metric-val { font-size: 1.8rem; font-weight: 800; color: #667eea; }
.metric-label { font-size: 0.78rem; color: #8892b0; margin-top: 0.2rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a1a 0%, #121228 100%);
    border-right: 1px solid rgba(102,126,234,0.1);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #ccd6f6 !important; font-weight: 600;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; padding: 8px 20px; font-weight: 600;
    background: rgba(102,126,234,0.08);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
}

/* Progress bars */
.score-bar-bg {
    width: 100%; height: 8px; border-radius: 4px;
    background: rgba(102,126,234,0.1); overflow: hidden;
}
.score-bar-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    transition: width 0.6s ease;
}

/* Algo card */
.algo-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid rgba(102,126,234,0.15); border-radius: 14px;
    padding: 1.5rem; margin-bottom: 1rem;
}
.algo-card h4 { color: #ccd6f6; margin-bottom: 0.5rem; }
.algo-card p { color: #8892b0; font-size: 0.88rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# ── Caching data & models ───────────────────────────────────────────
@st.cache_resource
def load_system():
    dm = DataManager(data_dir="data")
    dm.load_data()

    cb = ContentBasedRecommender()
    cb.fit(dm.movies)

    cf = CollaborativeRecommender(n_factors=15)
    cf.fit(dm.ratings, dm.movies)

    return dm, cb, cf


dm, cb_rec, cf_rec = load_system()
hybrid_rec = HybridRecommender(cb_rec, cf_rec)

GENRE_COLORS = {
    "Action": "#e63946", "Adventure": "#f4a261", "Animation": "#2ec4b6",
    "Biography": "#e9c46a", "Comedy": "#ffbe0b", "Crime": "#6c757d",
    "Drama": "#457b9d", "Family": "#80ed99", "Fantasy": "#c77dff",
    "History": "#bc6c25", "Horror": "#d00000", "Music": "#7209b7",
    "Musical": "#f72585", "Mystery": "#4361ee", "Romance": "#ff6b6b",
    "Sci-Fi": "#4cc9f0", "Thriller": "#ff9f1c", "War": "#606c38",
    "Western": "#dda15e", "Documentary": "#588157",
}


# ── Helpers ──────────────────────────────────────────────────────────
def genre_pills(genres_str):
    pills = []
    for g in genres_str.split("|"):
        g = g.strip()
        c = GENRE_COLORS.get(g, "#667eea")
        pills.append(
            f'<span class="genre-pill" style="color:{c};'
            f'border-color:{c}40;background:{c}18">{g}</span>'
        )
    return " ".join(pills)


def render_movie_card(movie, score=None, reason=None, rank=None):
    rank_badge = f'<span style="position:absolute;top:12px;right:14px;'\
        f'background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;'\
        f'padding:2px 10px;border-radius:12px;font-size:0.72rem;'\
        f'font-weight:700;">#{rank}</span>' if rank else ""

    score_html = ""
    if score is not None:
        pct = min(int(score * 100), 100)
        score_html = f"""
        <div style="margin-top:0.5rem;">
            <span class="score-badge">Match: {pct}%</span>
            <div class="score-bar-bg" style="margin-top:6px;">
                <div class="score-bar-fill" style="width:{pct}%"></div>
            </div>
        </div>"""

    reason_html = ""
    if reason:
        reason_html = f'<div class="reason-text">💡 {reason}</div>'

    m = movie
    overview = str(m.get("overview", ""))[:180]
    if len(str(m.get("overview", ""))) > 180:
        overview += "…"

    st.markdown(f"""
    <div class="movie-card">
        {rank_badge}
        <div class="movie-title">{m['title']}</div>
        <div class="movie-year">📅 {m['year']}  •  🎬 {m.get('director','')}</div>
        <div style="margin:0.5rem 0">{genre_pills(m['genres'])}</div>
        <div class="movie-overview">{overview}</div>
        {score_html}
        {reason_html}
    </div>""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
        <div style="font-size:2.5rem">🎬</div>
        <div style="font-size:1.3rem;font-weight:800;
            background:linear-gradient(135deg,#667eea,#764ba2);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            CineMatch</div>
        <div style="font-size:0.75rem;color:#8892b0;margin-top:2px">
            Hybrid Recommendation Engine</div>
    </div>
    <hr style="border-color:rgba(102,126,234,0.15);margin:0.8rem 0">
    """, unsafe_allow_html=True)

    users = dm.get_all_users()
    user_id = st.selectbox("👤 Select User", users,
                           format_func=lambda x: f"User {x}")

    algo = st.selectbox("🧠 Algorithm", [
        "Hybrid (Recommended)", "Content-Based Filtering",
        "Collaborative Filtering"
    ])

    n_recs = st.slider("📊 Number of Recommendations", 3, 15, 8)

    if "Hybrid" in algo:
        alpha = st.slider(
            "⚖️ Content ↔ Collaborative Weight",
            0.0, 1.0, 0.5, 0.05,
            help="0 = pure collaborative, 1 = pure content-based"
        )
    else:
        alpha = 0.5

    st.markdown('<hr style="border-color:rgba(102,126,234,0.15)">', unsafe_allow_html=True)

    # Quick stats
    u_ratings = dm.get_user_ratings(user_id)
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{len(u_ratings)}</div>
        <div class="metric-label">Movies Rated</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    avg_r = round(u_ratings["rating"].mean(), 1) if not u_ratings.empty else 0
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{avg_r} ★</div>
        <div class="metric-label">Average Rating</div>
    </div>""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">Your Personalised Picks</div>
<div class="hero-sub">
    Powered by TF-IDF content analysis &amp; SVD collaborative filtering
    — every suggestion comes with an explanation.
</div>""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 For You", "🔍 Explore", "📈 Analytics", "⚙️ How It Works"
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Personalised Recommendations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    # Get recommendations
    if "Hybrid" in algo:
        hybrid_rec.alpha = alpha
        recs = hybrid_rec.recommend_for_user(
            user_id, dm.ratings, dm.movies, n=n_recs
        )
    elif "Content" in algo:
        recs = cb_rec.recommend_for_user(
            user_id, dm.ratings, dm.movies, n=n_recs
        )
    else:
        recs = cf_rec.recommend_for_user(user_id, n=n_recs)

    if not recs:
        st.info("Not enough ratings to generate recommendations. "
                "Try another user.")
    else:
        cols = st.columns(2)
        for i, rec in enumerate(recs):
            movie = dm.get_movie_by_id(rec["movie_id"])
            with cols[i % 2]:
                render_movie_card(
                    movie, score=rec.get("score"),
                    reason=rec.get("reason"), rank=i + 1
                )

        # Score breakdown for hybrid
        if "Hybrid" in algo and recs:
            st.markdown("### 📊 Score Breakdown")
            breakdown = pd.DataFrame([{
                "Movie": r["title"],
                "Content Score": r.get("content_score", 0),
                "Collab Score": r.get("collab_score", 0),
                "Hybrid Score": r["score"],
            } for r in recs])

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Content-Based", x=breakdown["Movie"],
                y=breakdown["Content Score"],
                marker_color="#667eea",
            ))
            fig.add_trace(go.Bar(
                name="Collaborative", x=breakdown["Movie"],
                y=breakdown["Collab Score"],
                marker_color="#764ba2",
            ))
            fig.update_layout(
                barmode="group", height=380,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6",
                legend=dict(orientation="h", y=1.12),
                xaxis=dict(tickangle=-30),
                margin=dict(t=40, b=80),
            )
            st.plotly_chart(fig, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Explore Movies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("### 🔍 Browse & Find Similar Movies")

    c1, c2 = st.columns([2, 1])
    with c1:
        search = st.text_input("🔎 Search movies", placeholder="Type a title…")
    with c2:
        genre_filter = st.selectbox("🎭 Filter by genre",
                                    ["All"] + dm.get_genre_list())

    filtered = dm.movies.copy()
    if search:
        filtered = filtered[
            filtered["title"].str.contains(search, case=False, na=False)
        ]
    if genre_filter != "All":
        filtered = filtered[
            filtered["genres"].str.contains(genre_filter, na=False)
        ]

    # User's ratings for context
    user_rated = dm.get_user_ratings(user_id)
    rated_map = dict(zip(user_rated["movie_id"], user_rated["rating"]))

    for _, movie in filtered.head(20).iterrows():
        user_r = rated_map.get(movie["movie_id"])
        badge = f" — ⭐ You rated: {user_r}" if user_r else ""
        with st.expander(f"🎬 {movie['title']} ({movie['year']}){badge}"):
            render_movie_card(movie)

            if st.button(f"Find similar →", key=f"sim_{movie['movie_id']}"):
                similar = cb_rec.get_similar_movies(movie["movie_id"], n=5)
                st.markdown("**Similar movies:**")
                for s in similar:
                    sm = dm.get_movie_by_id(s["movie_id"])
                    render_movie_card(
                        sm, score=s["similarity_score"],
                        reason=s["reason"]
                    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — Analytics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("### 📈 Recommendation Analytics")

    # Metrics row
    metrics = cf_rec.evaluate()
    mc = st.columns(4)
    items = [
        ("🎯 RMSE", f"{metrics['rmse']:.3f}"),
        ("📏 MAE", f"{metrics['mae']:.3f}"),
        ("🧪 Test Samples", str(metrics["n_samples"])),
        ("🎬 Catalogue", str(len(dm.movies))),
    ]
    for col, (label, val) in zip(mc, items):
        col.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── User genre profile (radar) ──
    a1, a2 = st.columns(2)

    with a1:
        st.markdown("#### 🎭 Your Genre Profile")
        profile = dm.get_user_profile(user_id)
        if profile:
            cats = list(profile.keys())
            vals = list(profile.values())
            fig = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself",
                fillcolor="rgba(102,126,234,0.15)",
                line=dict(color="#667eea", width=2),
                marker=dict(size=6, color="#764ba2"),
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(range=[0, 5], showticklabels=True,
                                    gridcolor="rgba(102,126,234,0.1)",
                                    color="#8892b0"),
                    angularaxis=dict(gridcolor="rgba(102,126,234,0.1)",
                                     color="#ccd6f6"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                height=400, margin=dict(t=30, b=30),
                font_color="#ccd6f6",
            )
            st.plotly_chart(fig, use_container_width=True)

    with a2:
        st.markdown("#### 📊 Rating Distribution")
        fig2 = px.histogram(
            dm.get_user_ratings(user_id), x="rating", nbins=9,
            color_discrete_sequence=["#667eea"],
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccd6f6", height=400,
            xaxis_title="Rating", yaxis_title="Count",
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Similarity heatmap (top-15 movies) ──
    st.markdown("#### 🔥 Content Similarity Heatmap (Top 15 Movies)")
    sim_mat = cb_rec.get_similarity_matrix()[:15, :15]
    labels = dm.movies["title"].values[:15]
    fig3 = px.imshow(
        sim_mat, x=labels, y=labels,
        color_continuous_scale=["#0a0a1a", "#667eea", "#764ba2"],
        aspect="auto",
    )
    fig3.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccd6f6",
        xaxis=dict(tickangle=-45),
        margin=dict(t=20, b=100),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Global genre popularity ──
    st.markdown("#### 🌍 Genre Popularity Across All Ratings")
    genre_pop = {}
    for _, row in dm.ratings.iterrows():
        m = dm.movies.loc[dm.movies["movie_id"] == row["movie_id"]]
        if m.empty:
            continue
        for g in m.iloc[0]["genres"].split("|"):
            g = g.strip()
            genre_pop[g] = genre_pop.get(g, 0) + 1
    gp_df = pd.DataFrame(
        sorted(genre_pop.items(), key=lambda x: x[1], reverse=True),
        columns=["Genre", "Ratings"],
    )
    fig4 = px.bar(
        gp_df, x="Genre", y="Ratings",
        color="Ratings",
        color_continuous_scale=["#667eea", "#764ba2"],
    )
    fig4.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccd6f6", height=380,
        margin=dict(t=20, b=60),
    )
    st.plotly_chart(fig4, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — How It Works
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("### ⚙️ Under the Hood")

    st.markdown("""
    <div class="algo-card">
        <h4>🔤 Content-Based Filtering</h4>
        <p>
        Extracts features from each movie's <b>genres, director, cast,
        and plot overview</b> using <b>TF-IDF vectorisation</b> (up to
        5 000 bi-gram features).  Pairwise <b>cosine similarity</b> is
        computed across the entire catalogue, so when a user rates a
        movie highly we can instantly surface the most similar titles
        they haven't seen yet.
        </p>
    </div>

    <div class="algo-card">
        <h4>👥 Collaborative Filtering (SVD)</h4>
        <p>
        Builds a <b>user–item rating matrix</b> and applies <b>truncated
        Singular Value Decomposition</b> (k=15 latent factors) to
        factorise it into lower-dimensional user and item embeddings.
        Missing entries are reconstructed to predict how much a user
        would enjoy an unseen movie — purely from the rating patterns
        of similar users.
        </p>
    </div>

    <div class="algo-card">
        <h4>⚖️ Hybrid Engine</h4>
        <p>
        The hybrid mode <b>normalises and linearly combines</b> the
        scores from both engines using an adjustable <b>α weight</b>.
        Setting α = 1 gives pure content-based, α = 0 gives pure
        collaborative, and values in between blend the two — mitigating
        the cold-start problem of collaborative filtering and the
        limited-scope problem of content-based filtering.
        </p>
    </div>

    <div class="algo-card">
        <h4>💡 Explainability</h4>
        <p>
        Every recommendation includes an <b>explanation</b>: content-
        based picks cite the similar movie and shared features;
        collaborative picks cite the similar users who rated the movie
        highly; hybrid picks show both signals and their respective
        scores.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🏗️ System Architecture")
    st.code("""
    ┌─────────────────────────────────────────────────┐
    │                   CineMatch                      │
    │                                                  │
    │  ┌──────────────┐     ┌──────────────────────┐  │
    │  │  movies.csv   │────▶ Content-Based Engine  │  │
    │  │  (metadata)   │     │ TF-IDF + Cosine Sim  │──┐
    │  └──────────────┘     └──────────────────────┘  │ │
    │                                                  │ │ ┌──────────┐
    │  ┌──────────────┐     ┌──────────────────────┐  │ ├▶│  Hybrid  │
    │  │  ratings.csv  │────▶ Collaborative Engine  │  │ │ │  Blend   │
    │  │  (user×item)  │     │ SVD (k=15 factors)   │──┘ │ │  (α)     │
    │  └──────────────┘     └──────────────────────┘    │ └──────────┘
    │                                                    │       │
    │                                          ┌────────┘       │
    │                                          ▼                ▼
    │                                   Ranked & Explained Picks
    └─────────────────────────────────────────────────────────────┘
    """, language=None)

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(102,126,234,0.1);margin-top:2rem">
<div style="text-align:center;color:#4a5568;font-size:0.75rem;padding:0.5rem 0 1rem">
    CineMatch Recommendation System &nbsp;•&nbsp; Built with Streamlit,
    scikit-learn &amp; SciPy &nbsp;•&nbsp; CodSoft AI Internship — Task 4
</div>
""", unsafe_allow_html=True)
