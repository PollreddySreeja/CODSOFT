"""
content_based.py
----------------
Content-Based Filtering engine using TF-IDF vectorisation on combined
movie features (genres, director, cast, overview) and cosine similarity.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """Recommends movies based on content feature similarity."""

    def __init__(self):
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.movies = None
        self.feature_names = None

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------
    def fit(self, movies_df):
        """Build TF-IDF vectors and precompute cosine similarity matrix."""
        self.movies = movies_df.copy().reset_index(drop=True)

        # Combine textual features into one document per movie
        self.movies["_features"] = (
            self.movies["genres"].str.replace("|", " ", regex=False) + " "
            + self.movies["director"].fillna("") + " "
            + self.movies["cast"].fillna("").str.replace("|", " ", regex=False)
            + " " + self.movies["overview"].fillna("")
        )

        tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = tfidf.fit_transform(self.movies["_features"])
        self.feature_names = tfidf.get_feature_names_out()

        self.cosine_sim = cosine_similarity(
            self.tfidf_matrix, self.tfidf_matrix
        )

    # ------------------------------------------------------------------
    # Item-to-item similarity
    # ------------------------------------------------------------------
    def get_similar_movies(self, movie_id, n=10):
        """Return the *n* most similar movies to *movie_id*."""
        idx = self.movies.loc[
            self.movies["movie_id"] == movie_id
        ].index[0]

        sim_scores = sorted(
            enumerate(self.cosine_sim[idx]),
            key=lambda x: x[1],
            reverse=True,
        )[1: n + 1]

        results = []
        for i, score in sim_scores:
            movie = self.movies.iloc[i]
            results.append({
                "movie_id": movie["movie_id"],
                "title": movie["title"],
                "similarity_score": round(score, 4),
                "reason": self._explain(idx, i),
            })
        return results

    # ------------------------------------------------------------------
    # User-level recommendations
    # ------------------------------------------------------------------
    def recommend_for_user(self, user_id, ratings_df, movies_df, n=10):
        """Aggregate content similarity from the user's liked movies."""
        user_ratings = ratings_df[ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            return []

        liked = user_ratings[user_ratings["rating"] >= 3.5]
        if liked.empty:
            liked = user_ratings.nlargest(3, "rating")

        rated_ids = set(user_ratings["movie_id"].values)
        score_map, reason_map = {}, {}

        for _, row in liked.iterrows():
            mid = row["movie_id"]
            match = self.movies.loc[self.movies["movie_id"] == mid]
            if match.empty:
                continue
            idx = match.index[0]
            weight = row["rating"] / 5.0

            for j, sim in enumerate(self.cosine_sim[idx]):
                tid = self.movies.iloc[j]["movie_id"]
                if tid in rated_ids:
                    continue
                w_score = sim * weight
                if tid not in score_map or w_score > score_map[tid]:
                    score_map[tid] = w_score
                    src_title = self.movies.iloc[idx]["title"]
                    reason_map[tid] = (
                        f"Similar to '{src_title}' "
                        f"(rated {row['rating']}★)"
                    )

        top = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:n]

        results = []
        for movie_id, score in top:
            movie = self.movies.loc[
                self.movies["movie_id"] == movie_id
            ].iloc[0]
            results.append({
                "movie_id": movie_id,
                "title": movie["title"],
                "score": round(score, 4),
                "reason": reason_map.get(movie_id, "Content match"),
            })
        return results

    # ------------------------------------------------------------------
    # Explanation helper
    # ------------------------------------------------------------------
    def _explain(self, idx1, idx2):
        """Human-readable explanation of why two movies are similar."""
        m1 = self.movies.iloc[idx1]
        m2 = self.movies.iloc[idx2]

        parts = []
        common_g = set(m1["genres"].split("|")) & set(m2["genres"].split("|"))
        if common_g:
            parts.append(f"Shared genres: {', '.join(common_g)}")
        if m1["director"] == m2["director"]:
            parts.append(f"Same director: {m1['director']}")

        c1 = set(str(m1["cast"]).split("|"))
        c2 = set(str(m2["cast"]).split("|"))
        common_c = c1 & c2
        if common_c:
            parts.append(
                f"Common cast: {', '.join(list(common_c)[:2])}"
            )
        return "; ".join(parts) if parts else "Thematic similarity"

    def get_similarity_matrix(self):
        return self.cosine_sim
