"""
data_manager.py
---------------
Handles loading movie metadata, generating synthetic user ratings,
and building the user-item interaction matrix used by both filtering engines.
"""

import pandas as pd
import numpy as np
import os


class DataManager:
    """Central data handler for the recommendation pipeline."""

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.movies = None
        self.ratings = None
        self.user_item_matrix = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_data(self):
        """Load movies CSV and either load or generate ratings."""
        movies_path = os.path.join(self.data_dir, "movies.csv")
        self.movies = pd.read_csv(movies_path)

        ratings_path = os.path.join(self.data_dir, "ratings.csv")
        if os.path.exists(ratings_path):
            self.ratings = pd.read_csv(ratings_path)
        else:
            self.ratings = self._generate_synthetic_ratings()
            self.ratings.to_csv(ratings_path, index=False)

        self._build_user_item_matrix()
        return self.movies, self.ratings

    # ------------------------------------------------------------------
    # Synthetic rating generation
    # ------------------------------------------------------------------
    def _generate_synthetic_ratings(self, n_users=30, sparsity=0.35):
        """
        Create realistic synthetic ratings by assigning each user a
        hidden genre-preference profile and sampling ratings accordingly.
        """
        np.random.seed(42)
        movie_ids = self.movies["movie_id"].values

        all_genres = set()
        for g in self.movies["genres"]:
            all_genres.update(g.split("|"))
        all_genres = sorted(all_genres)

        # Build latent user profiles
        user_profiles = {}
        for uid in range(1, n_users + 1):
            prefs = {g: np.random.uniform(0.2, 0.8) for g in all_genres}
            favourites = np.random.choice(
                all_genres, size=np.random.randint(2, 5), replace=False
            )
            for g in favourites:
                prefs[g] = min(1.0, prefs[g] + 0.4)
            user_profiles[uid] = prefs

        rows = []
        for uid in range(1, n_users + 1):
            n_rated = int(len(movie_ids) * np.random.uniform(
                sparsity - 0.1, sparsity + 0.15
            ))
            rated = np.random.choice(movie_ids, size=n_rated, replace=False)

            for mid in rated:
                movie = self.movies.loc[self.movies["movie_id"] == mid].iloc[0]
                m_genres = movie["genres"].split("|")
                genre_score = np.mean(
                    [user_profiles[uid].get(g, 0.5) for g in m_genres]
                )
                base = genre_score * 4 + 1          # map to 1-5 range
                noise = np.random.normal(0, 0.5)
                rating = np.clip(round(base + noise, 1), 1.0, 5.0)
                rating = round(rating * 2) / 2      # snap to nearest 0.5

                rows.append({
                    "user_id": uid,
                    "movie_id": int(mid),
                    "rating": rating,
                    "timestamp": int(np.random.uniform(1.6e9, 1.7e9)),
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Matrix construction
    # ------------------------------------------------------------------
    def _build_user_item_matrix(self):
        self.user_item_matrix = self.ratings.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        ).fillna(0)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_user_ratings(self, user_id):
        return self.ratings[self.ratings["user_id"] == user_id]

    def get_unrated_movies(self, user_id):
        rated_ids = self.get_user_ratings(user_id)["movie_id"].values
        return self.movies[~self.movies["movie_id"].isin(rated_ids)]

    def get_all_users(self):
        return sorted(self.ratings["user_id"].unique())

    def get_movie_by_id(self, movie_id):
        return self.movies.loc[self.movies["movie_id"] == movie_id].iloc[0]

    def get_user_profile(self, user_id):
        """Compute average rating per genre for a user."""
        user_ratings = self.get_user_ratings(user_id)
        if user_ratings.empty:
            return {}

        genre_totals, genre_counts = {}, {}
        for _, row in user_ratings.iterrows():
            movie = self.movies.loc[
                self.movies["movie_id"] == row["movie_id"]
            ].iloc[0]
            for g in movie["genres"].split("|"):
                g = g.strip()
                genre_totals[g] = genre_totals.get(g, 0) + row["rating"]
                genre_counts[g] = genre_counts.get(g, 0) + 1

        return {g: round(genre_totals[g] / genre_counts[g], 2)
                for g in genre_totals}

    def get_genre_list(self):
        """Return sorted list of all unique genres in the dataset."""
        genres = set()
        for g in self.movies["genres"]:
            genres.update(g.split("|"))
        return sorted(genres)
