"""
FaceVision AI - Face Recognition Engine
Uses DeepFace with ArcFace model for state-of-the-art face recognition.
Supports face registration, identification, and verification.
"""

import os
import json
import shutil
import time
import cv2
import numpy as np


class FaceRecognizer:
    """
    Face recognition system using DeepFace with ArcFace embedding model.
    Manages a face database for registration and identification.
    """

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
        self.db_path = db_path
        self.known_faces_dir = os.path.join(db_path, 'known_faces')
        self.metadata_path = os.path.join(db_path, 'metadata.json')
        os.makedirs(self.known_faces_dir, exist_ok=True)

        self.model_name = 'ArcFace'
        self.detector_backend = 'opencv'
        self._deepface = None
        self._load_metadata()

    def _ensure_deepface(self):
        """Lazy-load DeepFace to avoid slow startup."""
        if self._deepface is None:
            from deepface import DeepFace
            self._deepface = DeepFace

    def _load_metadata(self):
        """Load face database metadata."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'people': {}, 'total_registrations': 0}

    def _save_metadata(self):
        """Save face database metadata."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    # ── Registration ──────────────────────────────────────────────────────────

    def register_face(self, image, name):
        """
        Register a face image under a given name.
        Multiple images per person are supported for better accuracy.
        """
        # Create person directory
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        person_dir = os.path.join(self.known_faces_dir, safe_name)
        os.makedirs(person_dir, exist_ok=True)

        # Save the image
        existing = len([f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png'))])
        img_filename = f"{safe_name}_{existing + 1}.jpg"
        img_path = os.path.join(person_dir, img_filename)
        cv2.imwrite(img_path, image)

        # Update metadata
        if safe_name not in self.metadata['people']:
            self.metadata['people'][safe_name] = {
                'name': name,
                'images': [],
                'registered_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        self.metadata['people'][safe_name]['images'].append(img_filename)
        self.metadata['total_registrations'] += 1
        self._save_metadata()

        # Clear any cached representations
        self._clear_representations_cache()

        return img_path

    def _clear_representations_cache(self):
        """Clear DeepFace's cached representations to force re-computation."""
        for root, dirs, files in os.walk(self.known_faces_dir):
            for f in files:
                if f.startswith('ds_model_') or f.endswith('.pkl'):
                    os.remove(os.path.join(root, f))

    # ── Recognition ───────────────────────────────────────────────────────────

    def recognize_face(self, image_path):
        """
        Recognize a face against the registered face database.
        Returns identified person name and confidence.
        """
        self._ensure_deepface()

        if not self.get_registered_people():
            return {'error': 'No faces registered in the database.'}

        try:
            results = self._deepface.find(
                img_path=image_path,
                db_path=self.known_faces_dir,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True
            )

            recognized = []
            for df in results:
                if len(df) > 0:
                    # Get the best match
                    best = df.iloc[0]
                    identity_path = best['identity']

                    # Extract person name from directory structure
                    rel_path = os.path.relpath(identity_path, self.known_faces_dir)
                    person_name = rel_path.split(os.sep)[0]

                    # Get display name from metadata
                    display_name = person_name
                    if person_name in self.metadata['people']:
                        display_name = self.metadata['people'][person_name]['name']

                    # Calculate similarity (lower distance = more similar)
                    distance_col = [c for c in df.columns if 'distance' in c.lower()]
                    distance = float(best[distance_col[0]]) if distance_col else 0
                    similarity = max(0, 1 - distance) * 100

                    recognized.append({
                        'name': display_name,
                        'distance': distance,
                        'similarity': similarity,
                        'identity_path': identity_path
                    })

            return {'matches': recognized}

        except Exception as e:
            return {'error': str(e)}

    # ── Verification ──────────────────────────────────────────────────────────

    def verify_faces(self, img1_path, img2_path):
        """
        Verify whether two face images belong to the same person.
        Returns verification result with distance and threshold.
        """
        self._ensure_deepface()

        try:
            result = self._deepface.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            return {
                'verified': result['verified'],
                'distance': result['distance'],
                'threshold': result['threshold'],
                'similarity': max(0, (1 - result['distance'])) * 100,
                'model': result['model'],
            }
        except Exception as e:
            return {'error': str(e)}

    # ── Database Management ───────────────────────────────────────────────────

    def get_registered_people(self):
        """Get dictionary of registered people and their image paths."""
        people = {}
        if os.path.exists(self.known_faces_dir):
            for person in sorted(os.listdir(self.known_faces_dir)):
                person_dir = os.path.join(self.known_faces_dir, person)
                if os.path.isdir(person_dir):
                    images = [
                        os.path.join(person_dir, f)
                        for f in sorted(os.listdir(person_dir))
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                    ]
                    if images:
                        display_name = person
                        if person in self.metadata.get('people', {}):
                            display_name = self.metadata['people'][person]['name']
                        people[display_name] = images
        return people

    def delete_person(self, name):
        """Remove a registered person and their images from the database."""
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        person_dir = os.path.join(self.known_faces_dir, safe_name)

        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
            if safe_name in self.metadata['people']:
                del self.metadata['people'][safe_name]
                self._save_metadata()
            self._clear_representations_cache()
            return True
        return False

    def get_stats(self):
        """Get database statistics."""
        people = self.get_registered_people()
        total_images = sum(len(imgs) for imgs in people.values())
        return {
            'total_people': len(people),
            'total_images': total_images,
            'total_registrations': self.metadata.get('total_registrations', 0)
        }
