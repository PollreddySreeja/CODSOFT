"""
FaceVision AI - Face Analysis Engine
Uses DeepFace to analyze facial attributes: age, gender, emotion, and ethnicity.
"""

import time


class FaceAnalyzer:
    """
    Analyzes facial attributes from images using DeepFace.
    Supports age estimation, gender classification, emotion recognition, and ethnicity analysis.
    """

    def __init__(self):
        self._deepface = None
        self.detector_backend = 'opencv'

    def _ensure_deepface(self):
        """Lazy-load DeepFace."""
        if self._deepface is None:
            from deepface import DeepFace
            self._deepface = DeepFace

    def analyze(self, image_path, actions=None):
        """
        Perform comprehensive facial analysis on an image.

        Args:
            image_path: Path to the image file.
            actions: List of analyses to perform. Options: 'age', 'gender', 'emotion', 'race'.
                     Defaults to all four.

        Returns:
            List of analysis results for each detected face, or dict with error key.
        """
        self._ensure_deepface()

        if actions is None:
            actions = ['age', 'gender', 'emotion', 'race']

        start = time.perf_counter()

        try:
            results = self._deepface.analyze(
                img_path=image_path,
                actions=actions,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True
            )
            elapsed = time.perf_counter() - start

            # Normalize results to always be a list
            if isinstance(results, dict):
                results = [results]

            # Enrich each result
            for r in results:
                r['processing_time'] = elapsed

            return results

        except Exception as e:
            return {'error': str(e)}

    def get_emotion_summary(self, analysis_result):
        """
        Extract and format emotion data from an analysis result.
        Returns sorted emotions with their confidence percentages.
        """
        if 'emotion' not in analysis_result:
            return []

        emotions = analysis_result['emotion']
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        return [
            {'emotion': name.capitalize(), 'confidence': score}
            for name, score in sorted_emotions
        ]

    def get_dominant_attributes(self, analysis_result):
        """
        Extract the dominant attribute for each analysis category.
        """
        summary = {}

        if 'age' in analysis_result:
            summary['age'] = analysis_result['age']

        if 'dominant_gender' in analysis_result:
            summary['gender'] = analysis_result['dominant_gender']
            if 'gender' in analysis_result:
                summary['gender_confidence'] = analysis_result['gender'].get(
                    analysis_result['dominant_gender'], 0
                )

        if 'dominant_emotion' in analysis_result:
            summary['emotion'] = analysis_result['dominant_emotion'].capitalize()
            if 'emotion' in analysis_result:
                summary['emotion_confidence'] = analysis_result['emotion'].get(
                    analysis_result['dominant_emotion'], 0
                )

        if 'dominant_race' in analysis_result:
            summary['ethnicity'] = analysis_result['dominant_race'].capitalize()
            if 'race' in analysis_result:
                summary['ethnicity_confidence'] = analysis_result['race'].get(
                    analysis_result['dominant_race'], 0
                )

        return summary
