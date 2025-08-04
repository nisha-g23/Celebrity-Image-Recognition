# -*- coding: utf-8 -*-
"""Enhanced Celebrity Image Recognition Model with Incremental Training

This model supports:
- Incremental training with new datasets
- Model persistence and loading
- Streamlit deployment
- Real-time face recognition
"""

# Install required packages
# !pip install insightface onnxruntime opencv-python scikit-learn matplotlib numpy pandas tqdm streamlit

# Import required libraries
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import insightface
from insightface.app import FaceAnalysis
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pickle
import json
import shutil
from datetime import datetime
import streamlit as st
from PIL import Image
import io

# Model persistence paths
MODEL_DIR = "saved_models"
EMBEDDINGS_FILE = os.path.join(MODEL_DIR, "embeddings.pkl")
LABEL_ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.pkl")
SVM_MODEL_FILE = os.path.join(MODEL_DIR, "svm_model.pkl")
MODEL_CONFIG_FILE = os.path.join(MODEL_DIR, "model_config.json")

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

class IncrementalCelebrityRecognizer:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.app = None
        self.svm_model = None
        self.label_encoder = None
        self.existing_embeddings = []
        self.existing_labels = []
        self.celebrities_info = {}
        
        # Initialize InsightFace
        self._initialize_insightface()
        
    def _initialize_insightface(self):
        """Initialize InsightFace application"""
        try:
            self.app = FaceAnalysis(name='buffalo_l')
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print("InsightFace initialized successfully!")
        except Exception as e:
            print(f"Error initializing InsightFace: {e}")
            raise
    
    def save_model(self):
        """Save the current model state"""
        try:
            # Save SVM model
            with open(SVM_MODEL_FILE, 'wb') as f:
                pickle.dump(self.svm_model, f)
            
            # Save label encoder
            with open(LABEL_ENCODER_FILE, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            # Save embeddings and labels
            embeddings_data = {
                'embeddings': self.existing_embeddings,
                'labels': self.existing_labels
            }
            with open(EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump(embeddings_data, f)
            
            # Save model configuration
            config = {
                'num_celebrities': len(self.label_encoder.classes_) if self.label_encoder else 0,
                'feature_dimension': len(self.existing_embeddings[0]) if self.existing_embeddings else 0,
                'last_updated': datetime.now().isoformat(),
                'celebrities': list(self.label_encoder.classes_) if self.label_encoder else []
            }
            with open(MODEL_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Model saved successfully to {self.model_dir}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self):
        """Load existing model if available"""
        try:
            if (os.path.exists(SVM_MODEL_FILE) and 
                os.path.exists(LABEL_ENCODER_FILE) and 
                os.path.exists(EMBEDDINGS_FILE)):
                
                # Load SVM model
                with open(SVM_MODEL_FILE, 'rb') as f:
                    self.svm_model = pickle.load(f)
                
                # Load label encoder
                with open(LABEL_ENCODER_FILE, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                
                # Load embeddings and labels
                with open(EMBEDDINGS_FILE, 'rb') as f:
                    embeddings_data = pickle.load(f)
                    self.existing_embeddings = embeddings_data['embeddings']
                    self.existing_labels = embeddings_data['labels']
                
                print(f"Model loaded successfully!")
                print(f"Loaded {len(self.label_encoder.classes_)} celebrities")
                print(f"Total embeddings: {len(self.existing_embeddings)}")
                return True
            else:
                print("No existing model found. Will train new model.")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def explore_dataset(self, dataset_path):
        """Explore the dataset structure and count images per celebrity"""
        celebrities = {}
        total_images = 0

        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                image_files = [f for f in os.listdir(item_path)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if len(image_files) > 0:
                    celebrities[item] = len(image_files)
                    total_images += len(image_files)

        print(f"Total celebrities found: {len(celebrities)}")
        print(f"Total images: {total_images}")
        print("\nCelebrities and their image counts:")
        for celeb, count in sorted(celebrities.items()):
            print(f"{celeb}: {count} images")

        self.celebrities_info = celebrities
        return celebrities

    def detect_and_extract_face(self, image_path):
        """Detect faces in an image and extract features"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, None, None

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.app.get(img_rgb)

            if len(faces) == 0:
                return None, None, None

            largest_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
            embedding = largest_face.embedding
            bbox = largest_face.bbox.astype(int)

            x1, y1, x2, y2 = bbox
            face_crop = img_rgb[y1:y2, x1:x2]

            return embedding, face_crop, bbox

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None, None, None

    def extract_dataset_features(self, dataset_path):
        """Extract face features from the dataset"""
        features = []
        labels = []
        face_crops = []

        print("Extracting features from dataset...")

        celebrity_folders = []
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                image_files = [f for f in os.listdir(item_path)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if len(image_files) > 0:
                    celebrity_folders.append(item)

        print(f"Found {len(celebrity_folders)} celebrity folders")

        for celeb_folder in tqdm(celebrity_folders):
            celeb_path = os.path.join(dataset_path, celeb_folder)
            image_files = [f for f in os.listdir(celeb_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            for image_file in image_files:
                image_path = os.path.join(celeb_path, image_file)
                embedding, face_crop, bbox = self.detect_and_extract_face(image_path)

                if embedding is not None:
                    features.append(embedding)
                    labels.append(celeb_folder)
                    face_crops.append(face_crop)

        print(f"Extracted features for {len(features)} faces from {len(set(labels))} celebrities")
        return np.array(features), np.array(labels), face_crops

    def incremental_train(self, new_dataset_path, test_size=0.2):
        """Incrementally train the model with new data"""
        print("Starting incremental training...")
        
        # Extract features from new dataset
        new_features, new_labels, _ = self.extract_dataset_features(new_dataset_path)
        
        if len(new_features) == 0:
            print("No valid faces found in new dataset!")
            return False
        
        # Combine with existing data if available
        if len(self.existing_embeddings) > 0:
            print("Combining with existing embeddings...")
            all_features = np.vstack([self.existing_embeddings, new_features])
            all_labels = np.concatenate([self.existing_labels, new_labels])
        else:
            print("No existing embeddings found. Training new model...")
            all_features = new_features
            all_labels = new_labels
        
        # Create or update label encoder
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(all_labels)
        else:
            # Add new labels to existing encoder
            existing_classes = set(self.label_encoder.classes_)
            new_classes = set(all_labels)
            all_classes = list(existing_classes.union(new_classes))
            
            # Create new encoder with all classes
            new_encoder = LabelEncoder()
            new_encoder.fit(all_classes)
            
            # Transform all labels
            y_encoded = new_encoder.transform(all_labels)
            self.label_encoder = new_encoder
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            all_features, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        print(f"Total celebrities: {len(self.label_encoder.classes_)}")
        
        # Train SVM classifier
        print("Training SVM classifier...")
        self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        self.svm_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        
        # Update stored embeddings and labels
        self.existing_embeddings = all_features
        self.existing_labels = all_labels
        
        # Save the updated model
        self.save_model()
        
        return True

    def recognize_face(self, image_path, confidence_threshold=0.5):
        """Recognize faces in an image"""
        if self.svm_model is None or self.label_encoder is None:
            print("Model not trained yet!")
            return None, None, None, None
        
        embedding, face_crop, bbox = self.detect_and_extract_face(image_path)
        
        if embedding is None:
            return None, None, None, None
        
        prediction = self.svm_model.predict([embedding])[0]
        confidence = np.max(self.svm_model.predict_proba([embedding]))
        celebrity_name = self.label_encoder.inverse_transform([prediction])[0]
        
        return celebrity_name, confidence, face_crop, bbox

    def get_model_info(self):
        """Get information about the current model"""
        if self.label_encoder is None:
            return {
                'trained': False,
                'num_celebrities': 0,
                'total_embeddings': 0,
                'celebrities': []
            }
        
        return {
            'trained': True,
            'num_celebrities': len(self.label_encoder.classes_),
            'total_embeddings': len(self.existing_embeddings),
            'celebrities': list(self.label_encoder.classes_)
        }
    
    def retrain_with_new_dataset(self, new_dataset_path, preserve_existing=True):
        """Retrain the model with a new dataset, optionally preserving existing data"""
        print("Starting dataset retraining...")
        
        if preserve_existing and len(self.existing_embeddings) > 0:
            print("Preserving existing embeddings and adding new data...")
            return self.incremental_train(new_dataset_path)
        else:
            print("Starting fresh training with new dataset...")
            # Clear existing data
            self.existing_embeddings = []
            self.existing_labels = []
            self.svm_model = None
            self.label_encoder = None
            return self.incremental_train(new_dataset_path)
    
    def add_single_celebrity(self, celebrity_name, image_paths):
        """Add a single celebrity with multiple images"""
        print(f"Adding celebrity: {celebrity_name}")
        
        features = []
        labels = []
        
        for image_path in image_paths:
            embedding, face_crop, bbox = self.detect_and_extract_face(image_path)
            if embedding is not None:
                features.append(embedding)
                labels.append(celebrity_name)
        
        if len(features) == 0:
            print("No valid faces found in the provided images!")
            return False
        
        # Convert to numpy arrays
        new_features = np.array(features)
        new_labels = np.array(labels)
        
        # Combine with existing data
        if len(self.existing_embeddings) > 0:
            all_features = np.vstack([self.existing_embeddings, new_features])
            all_labels = np.concatenate([self.existing_labels, new_labels])
        else:
            all_features = new_features
            all_labels = new_labels
        
        # Update label encoder
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(all_labels)
        else:
            existing_classes = set(self.label_encoder.classes_)
            new_classes = set(all_labels)
            all_classes = list(existing_classes.union(new_classes))
            
            new_encoder = LabelEncoder()
            new_encoder.fit(all_classes)
            y_encoded = new_encoder.transform(all_labels)
            self.label_encoder = new_encoder
        
        # Train SVM
        self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        self.svm_model.fit(all_features, y_encoded)
        
        # Update stored data
        self.existing_embeddings = all_features
        self.existing_labels = all_labels
        
        # Save model
        self.save_model()
        
        print(f"Successfully added {celebrity_name} with {len(features)} images")
        return True
    
    def remove_celebrity(self, celebrity_name):
        """Remove a celebrity from the model"""
        if self.label_encoder is None:
            print("No model to modify!")
            return False
        
        if celebrity_name not in self.label_encoder.classes_:
            print(f"Celebrity '{celebrity_name}' not found in model!")
            return False
        
        # Find indices to keep
        keep_indices = [i for i, label in enumerate(self.existing_labels) if label != celebrity_name]
        
        if len(keep_indices) == 0:
            print("Cannot remove all celebrities!")
            return False
        
        # Filter data
        self.existing_embeddings = self.existing_embeddings[keep_indices]
        self.existing_labels = [self.existing_labels[i] for i in keep_indices]
        
        # Update label encoder
        remaining_classes = list(set(self.existing_labels))
        new_encoder = LabelEncoder()
        new_encoder.fit(remaining_classes)
        y_encoded = new_encoder.transform(self.existing_labels)
        self.label_encoder = new_encoder
        
        # Retrain SVM
        self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        self.svm_model.fit(self.existing_embeddings, y_encoded)
        
        # Save model
        self.save_model()
        
        print(f"Successfully removed {celebrity_name} from model")
        return True
    
    def get_training_history(self):
        """Get training history and statistics"""
        if not os.path.exists(MODEL_CONFIG_FILE):
            return None
        
        try:
            with open(MODEL_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading training history: {e}")
            return None

# Streamlit App
def create_streamlit_app():
    """Create the Streamlit application"""
    
    st.set_page_config(
        page_title="Celebrity Face Recognition",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("🎭 Celebrity Face Recognition System")
    st.markdown("---")
    
    # Initialize recognizer
    if 'recognizer' not in st.session_state:
        st.session_state.recognizer = IncrementalCelebrityRecognizer()
        st.session_state.recognizer.load_model()
    
    recognizer = st.session_state.recognizer
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Face Recognition", "Model Training", "Model Info"]
    )
    
    if page == "Face Recognition":
        st.header("🔍 Face Recognition")
        
        # Model status
        model_info = recognizer.get_model_info()
        if model_info['trained']:
            st.success(f"✅ Model trained with {model_info['num_celebrities']} celebrities")
        else:
            st.warning("⚠️ Model not trained yet. Please train the model first.")
            return
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload an image to recognize celebrities",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Recognition Results")
                
                # Recognize faces
                celebrity_name, confidence, face_crop, bbox = recognizer.recognize_face("temp_image.jpg")
                
                if celebrity_name is not None:
                    st.success(f"🎯 Recognized: **{celebrity_name}**")
                    st.info(f"Confidence: **{confidence:.2f}**")
                    
                    if face_crop is not None:
                        st.subheader("Detected Face")
                        face_img = Image.fromarray(face_crop)
                        st.image(face_img, use_column_width=True)
                else:
                    st.error("❌ No face detected in the image")
            
            # Clean up
            if os.path.exists("temp_image.jpg"):
                os.remove("temp_image.jpg")
    
    elif page == "Model Training":
        st.header("🏋️ Model Training")
        
        # Model info
        model_info = recognizer.get_model_info()
        st.info(f"Current model: {model_info['num_celebrities']} celebrities, {model_info['total_embeddings']} embeddings")
        
        # Training options
        st.subheader("Training Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("New Training")
            new_dataset_path = st.text_input(
                "Enter path to new dataset folder:",
                placeholder="e.g., /path/to/celebrity/dataset"
            )
            
            if st.button("Train New Model") and new_dataset_path:
                if os.path.exists(new_dataset_path):
                    with st.spinner("Training new model..."):
                        success = recognizer.incremental_train(new_dataset_path)
                        if success:
                            st.success("✅ Model trained successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Training failed!")
                else:
                    st.error("❌ Dataset path not found!")
        
        with col2:
            st.subheader("Incremental Training")
            incremental_dataset_path = st.text_input(
                "Enter path to additional dataset:",
                placeholder="e.g., /path/to/additional/celebrities"
            )
            
            if st.button("Add to Existing Model") and incremental_dataset_path:
                if os.path.exists(incremental_dataset_path):
                    with st.spinner("Adding to existing model..."):
                        success = recognizer.incremental_train(incremental_dataset_path)
                        if success:
                            st.success("✅ Model updated successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Update failed!")
                else:
                    st.error("❌ Dataset path not found!")
        
        # Dataset upload (for demo purposes)
        st.subheader("Upload Dataset")
        st.info("Upload a zip file containing celebrity folders with images")
        
        uploaded_dataset = st.file_uploader(
            "Upload dataset zip file",
            type=['zip']
        )
        
        if uploaded_dataset is not None:
            # Extract and process uploaded dataset
            st.info("Dataset upload functionality would be implemented here")
    
    elif page == "Model Info":
        st.header("📊 Model Information")
        
        model_info = recognizer.get_model_info()
        
        if model_info['trained']:
            st.success("✅ Model is trained and ready")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Number of Celebrities", model_info['num_celebrities'])
                st.metric("Total Embeddings", model_info['total_embeddings'])
            
            with col2:
                st.subheader("Recognized Celebrities")
                for i, celeb in enumerate(model_info['celebrities'], 1):
                    st.write(f"{i}. {celeb}")
            
            # Model performance metrics (if available)
            st.subheader("Model Performance")
            st.info("Performance metrics would be displayed here")
            
        else:
            st.warning("⚠️ Model is not trained yet")
            st.info("Please train the model using the Model Training page")

# Main execution
if __name__ == "__main__":
    # Initialize the recognizer
    recognizer = IncrementalCelebrityRecognizer()
    
    # Try to load existing model
    model_loaded = recognizer.load_model()
    
    if not model_loaded:
        print("No existing model found. You can train a new model using:")
        print("recognizer.incremental_train('/path/to/your/dataset')")
    
    # Example usage:
    # recognizer.incremental_train('/path/to/celebrity/dataset')
    # recognizer.save_model()
    
    print("\n" + "="*50)
    print("STREAMLIT DEPLOYMENT STEPS:")
    print("="*50)
    print("1. Install Streamlit: pip install streamlit")
    print("2. Create a new file called 'app.py' with the Streamlit code")
    print("3. Run the app: streamlit run app.py")
    print("4. Open your browser to the provided URL")
    print("\nThe app will provide:")
    print("- Real-time face recognition")
    print("- Model training interface")
    print("- Model information dashboard")
    print("="*50)

