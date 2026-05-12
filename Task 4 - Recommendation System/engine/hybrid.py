"""
hybrid.py
---------
Hybrid recommender that blends Content-Based and Collaborative Filtering
scores using an adjustable weighting parameter (alpha).
"""


class HybridRecommender:
    """Weighted hybrid of content-based and collaborative recommenders."""

    def __init__(self, content_rec, collab_rec, alpha=0.5):
        """
        Parameters
        ----------
        content_rec : ContentBasedRecommender
        collab_rec  : CollaborativeRecommender
        alpha       : float  (0 → pure collaborative, 1 → pure content)
        """
        self.content_rec = content_rec
        self.collab_rec = collab_rec
        self.alpha = alpha

    # ------------------------------------------------------------------
    def recommend_for_user(self, user_id, ratings_df, movies_df, n=10):
        """Merge and re-rank recommendations from both engines."""
        cb_recs = self.content_rec.recommend_for_user(
            user_id, ratings_df, movies_df, n=n * 2
        )
        cf_recs = self.collab_rec.recommend_for_user(user_id, n=n * 2)

        # Normalise scores to [0, 1]
        cb_scores, cb_reasons = self._normalise(cb_recs)
        cf_scores, cf_reasons = self._normalise(cf_recs)

        all_ids = set(cb_scores) | set(cf_scores)
        combined = {}

        for mid in all_ids:
            cs = cb_scores.get(mid, 0)
            cfs = cf_scores.get(mid, 0)
            hybrid = self.alpha * cs + (1 - self.alpha) * cfs

            reasons = []
            if mid in cb_reasons:
                reasons.append(f"[Content] {cb_reasons[mid]}")
            if mid in cf_reasons:
                reasons.append(f"[Collab] {cf_reasons[mid]}")

            combined[mid] = {
                "score": hybrid,
                "content_score": cs,
                "collab_score": cfs,
                "reason": " | ".join(reasons) or "Hybrid blend",
            }

        ranked = sorted(
            combined.items(), key=lambda x: x[1]["score"], reverse=True
        )[:n]

        results = []
        for movie_id, data in ranked:
            movie = movies_df.loc[movies_df["movie_id"] == movie_id]
            if movie.empty:
                continue
            movie = movie.iloc[0]
            results.append({
                "movie_id": movie_id,
                "title": movie["title"],
                "score": round(data["score"], 4),
                "content_score": round(data["content_score"], 4),
                "collab_score": round(data["collab_score"], 4),
                "reason": data["reason"],
            })
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(recs):
        if not recs:
            return {}, {}
        mx = max(r["score"] for r in recs) or 1
        scores = {r["movie_id"]: r["score"] / mx for r in recs}
        reasons = {r["movie_id"]: r["reason"] for r in recs}
        return scores, reasons
