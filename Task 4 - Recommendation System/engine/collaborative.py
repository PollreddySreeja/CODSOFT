"""
collaborative.py
-----------------
Collaborative Filtering engine using Singular Value Decomposition (SVD)
on the user-item rating matrix.  Predicts unobserved ratings and finds
similar users via cosine similarity in the latent factor space.
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeRecommender:
    """Matrix-factorisation recommender (truncated SVD)."""

    def __init__(self, n_factors=15):
        self.n_factors = n_factors
        self.user_item_matrix = None
        self.predicted_ratings = None
        self.user_similarity = None
        self.user_means = None
        self.movies = None
        self.ratings = None

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------
    def fit(self, ratings_df, movies_df):
        """Build the SVD model and predict all missing ratings."""
        self.movies = movies_df
        self.ratings = ratings_df

        # Pivot into user × movie matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        ).fillna(0)

        # Mean-centre per user
        self.user_means = self.user_item_matrix.mean(axis=1)
        normalised = self.user_item_matrix.sub(self.user_means, axis=0)

        # Truncated SVD
        k = min(self.n_factors, min(normalised.shape) - 1)
        U, sigma, Vt = svds(normalised.values.astype(float), k=k)

        # Reconstruct full prediction matrix
        predicted = np.dot(np.dot(U, np.diag(sigma)), Vt)
        self.predicted_ratings = pd.DataFrame(
            predicted,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns,
        )
        self.predicted_ratings = self.predicted_ratings.add(
            self.user_means, axis=0
        )

        # User-user cosine similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    def recommend_for_user(self, user_id, n=10):
        """Top-*n* predicted movies the user hasn't rated yet."""
        if user_id not in self.predicted_ratings.index:
            return []

        preds = self.predicted_ratings.loc[user_id]
        already = self.user_item_matrix.loc[user_id]
        candidates = preds[already == 0].sort_values(ascending=False).head(n)

        results = []
        for movie_id, pred in candidates.items():
            movie = self.movies.loc[self.movies["movie_id"] == movie_id]
            if movie.empty:
                continue
            movie = movie.iloc[0]
            results.append({
                "movie_id": movie_id,
                "title": movie["title"],
                "predicted_rating": round(float(np.clip(pred, 1, 5)), 2),
                "score": round(float(np.clip(pred / 5.0, 0, 1)), 4),
                "reason": self._peer_explanation(user_id, movie_id),
            })
        return results

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------
    def _peer_explanation(self, user_id, movie_id, top_k=3):
        """Find similar users who enjoyed this movie."""
        if user_id not in self.user_item_matrix.index:
            return "Predicted from latent factors"

        uid_idx = list(self.user_item_matrix.index).index(user_id)
        sim = self.user_similarity[uid_idx]
        order = np.argsort(sim)[::-1][1:]  # skip self

        peers = []
        for idx in order[:15]:
            other = self.user_item_matrix.index[idx]
            if movie_id in self.user_item_matrix.columns:
                r = self.user_item_matrix.loc[other, movie_id]
                if r > 0:
                    peers.append((other, r, sim[idx]))
            if len(peers) >= top_k:
                break

        if peers:
            bits = [
                f"User {u} rated {r}★ ({s:.0%} similar)"
                for u, r, s in peers
            ]
            return "Loved by similar users — " + "; ".join(bits[:2])
        return "Predicted via SVD matrix factorisation"

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, test_ratio=0.2):
        """Compute RMSE and MAE on a random held-out split."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        np.random.seed(42)
        mask = np.random.random(self.user_item_matrix.shape) < test_ratio
        observed = self.user_item_matrix.values > 0
        test_mask = mask & observed

        actual = self.user_item_matrix.values[test_mask]
        predicted = np.clip(self.predicted_ratings.values[test_mask], 1, 5)

        if len(actual) == 0:
            return {"rmse": 0, "mae": 0, "n_samples": 0}

        return {
            "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
            "mae": round(float(mean_absolute_error(actual, predicted)), 4),
            "n_samples": int(len(actual)),
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_user_similarity_matrix(self):
        return self.user_similarity

    def get_predicted_ratings(self):
        return self.predicted_ratings
