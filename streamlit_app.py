# -*- coding: utf-8 -*-
"""
Streamlit App for Celebrity Face Recognition
Deployment file for the enhanced celebrity recognition model
"""

import streamlit as st
import os
import sys
import tempfile
import zipfile
from PIL import Image
import numpy as np
import cv2
from streamlit_extras.switch_page_button import switch_page

if st.button("Go to Detection Page"):
    switch_page("🔍 Face Recognition")

# Add the current directory to path to import the main model
sys.path.append(os.path.dirname(os.path.abspath("C:/Users/NISHA GOSWAMI/Downloads/ImageRecognitionSystem.py")))

# Import the enhanced recognizer
from ImageRecognitionSystem import IncrementalCelebrityRecognizer

def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Celebrity Face Recognition",
        page_icon="👤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎭 Celebrity Face Recognition System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize recognizer in session state
    if 'recognizer' not in st.session_state:
        with st.spinner("Initializing face recognition model..."):
            st.session_state.recognizer = IncrementalCelebrityRecognizer()
            st.session_state.recognizer.load_model()
    
    recognizer = st.session_state.recognizer
    
    # Sidebar navigation
    st.sidebar.title("🚀 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🔍 Face Recognition", "🏋️ Model Training", "📊 Model Info", "⚙️ Settings"]
    )
    
    if page == "🏠 Home":
        show_home_page(recognizer)
    elif page == "🔍 Face Recognition":
        show_recognition_page(recognizer)
    elif page == "🏋️ Model Training":
        show_training_page(recognizer)
    elif page == "📊 Model Info":
        show_model_info_page(recognizer)
    elif page == "⚙️ Settings":
        show_settings_page(recognizer)

def show_home_page(recognizer):
    """Display the home page"""
    st.header("🏠 Welcome to Celebrity Face Recognition")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What this system can do:
        
        **🔍 Real-time Face Recognition**
        - Upload images and instantly recognize celebrities
        - High-accuracy face detection and recognition
        - Confidence scoring for predictions
        
        **🏋️ Incremental Learning**
        - Add new celebrities to existing model
        - Retrain with new datasets without losing previous knowledge
        - Persistent model storage and loading
        
        **📊 Model Management**
        - Monitor model performance
        - View recognized celebrities
        - Track model statistics
        """)
    
    with col2:
        # Model status card
        model_info = recognizer.get_model_info()
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("📊 Model Status")
        
        if model_info['trained']:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.write("✅ **Model is trained and ready**")
            st.write(f"**Celebrities:** {model_info['num_celebrities']}")
            st.write(f"**Embeddings:** {model_info['total_embeddings']}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.write("⚠️ **Model needs training**")
            st.write("Please train the model first")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Start Recognition", use_container_width=True):
            st.switch_page("🔍 Face Recognition")
    
    with col2:
        if st.button("🏋️ Train Model", use_container_width=True):
            st.switch_page("🏋️ Model Training")
    
    with col3:
        if st.button("📊 View Model Info", use_container_width=True):
            st.switch_page("📊 Model Info")

def show_recognition_page(recognizer):
    """Display the face recognition page"""
    st.header("🔍 Face Recognition")
    
    # Model status check
    model_info = recognizer.get_model_info()
    if not model_info['trained']:
        st.error("❌ Model is not trained yet. Please train the model first.")
        st.info("Go to the 'Model Training' page to train your model.")
        return
    
    st.success(f"✅ Model ready with {model_info['num_celebrities']} celebrities")
    
    # Recognition interface
    st.subheader("📸 Upload Image for Recognition")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image containing faces to recognize celebrities"
        )
    
    with col2:
        st.markdown("""
        **Supported formats:** JPG, JPEG, PNG, BMP
        
        **Tips for better recognition:**
        - Use clear, well-lit images
        - Ensure faces are clearly visible
        - Avoid heavily edited or filtered images
        """)
    
    if uploaded_file is not None:
        # Process the uploaded image
        with st.spinner("Processing image..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name
            
            try:
                # Display original image
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📷 Original Image")
                    image = Image.open(uploaded_file)
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("🎯 Recognition Results")
                    
                    # Recognize faces
                    celebrity_name, confidence, face_crop, bbox = recognizer.recognize_face(temp_path)
                    
                    if celebrity_name is not None:
                        # Display results
                        if confidence >= 0.7:
                            st.success(f"🎯 **Recognized:** {celebrity_name}")
                        elif confidence >= 0.5:
                            st.warning(f"🤔 **Possible:** {celebrity_name}")
                        else:
                            st.error(f"❓ **Low confidence:** {celebrity_name}")
                        
                        st.info(f"**Confidence:** {confidence:.2%}")
                        
                        # Show confidence bar
                        st.progress(confidence)
                        
                        # Display cropped face if available
                        if face_crop is not None:
                            st.subheader("👤 Detected Face")
                            face_img = Image.fromarray(face_crop)
                            st.image(face_img, use_column_width=True)
                        
                        # Additional info
                        st.markdown(f"""
                        **Details:**
                        - **Celebrity:** {celebrity_name}
                        - **Confidence:** {confidence:.2%}
                        - **Face detected:** ✅
                        """)
                    else:
                        st.error("❌ No face detected in the image")
                        st.info("Please try an image with clearer faces")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

def show_training_page(recognizer):
    """Display the model training page"""
    st.header("🏋️ Model Training")
    
    # Current model info
    model_info = recognizer.get_model_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Current Model Status")
        if model_info['trained']:
            st.success(f"✅ Model trained with {model_info['num_celebrities']} celebrities")
            st.info(f"Total embeddings: {model_info['total_embeddings']}")
        else:
            st.warning("⚠️ No trained model found")
    
    with col2:
        st.subheader("📈 Training Options")
        st.info("Choose your training method below")
    
    # Training methods
    st.subheader("🎯 Training Methods")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🆕 New Training", "➕ Incremental Training", "🔄 Retrain Dataset", "👤 Add Single Celebrity", "🗑️ Remove Celebrity"])
    
    with tab1:
        st.markdown("**Train a completely new model**")
        new_dataset_path = st.text_input(
            "Enter path to dataset folder:",
            placeholder="e.g., C:/datasets/celebrities"
        )
        
        preserve_existing = st.checkbox("Preserve existing data (if any)", value=False)
        
        if st.button("🚀 Start New Training", type="primary"):
            if new_dataset_path and os.path.exists(new_dataset_path):
                with st.spinner("Training new model..."):
                    try:
                        success = recognizer.retrain_with_new_dataset(new_dataset_path, preserve_existing)
                        if success:
                            st.success("✅ Model trained successfully!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Training failed!")
                    except Exception as e:
                        st.error(f"❌ Error during training: {str(e)}")
            else:
                st.error("❌ Please enter a valid dataset path")
    
    with tab2:
        st.markdown("**Add new celebrities to existing model**")
        incremental_dataset_path = st.text_input(
            "Enter path to additional dataset:",
            placeholder="e.g., C:/datasets/new_celebrities"
        )
        
        if st.button("➕ Add to Existing Model"):
            if incremental_dataset_path and os.path.exists(incremental_dataset_path):
                with st.spinner("Adding to existing model..."):
                    try:
                        success = recognizer.incremental_train(incremental_dataset_path)
                        if success:
                            st.success("✅ Model updated successfully!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Update failed!")
                    except Exception as e:
                        st.error(f"❌ Error during update: {str(e)}")
            else:
                st.error("❌ Please enter a valid dataset path")
    
    with tab3:
        st.markdown("**Retrain with new dataset**")
        retrain_dataset_path = st.text_input(
            "Enter path to new dataset:",
            placeholder="e.g., C:/datasets/updated_celebrities"
        )
        
        preserve_existing = st.checkbox("Preserve existing embeddings", value=True)
        
        if st.button("🔄 Retrain Model"):
            if retrain_dataset_path and os.path.exists(retrain_dataset_path):
                with st.spinner("Retraining model..."):
                    try:
                        success = recognizer.retrain_with_new_dataset(retrain_dataset_path, preserve_existing)
                        if success:
                            st.success("✅ Model retrained successfully!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Retraining failed!")
                    except Exception as e:
                        st.error(f"❌ Error during retraining: {str(e)}")
            else:
                st.error("❌ Please enter a valid dataset path")
    
    with tab4:
        st.markdown("**Add a single celebrity with multiple images**")
        
        celebrity_name = st.text_input("Celebrity name:", placeholder="e.g., Tom Hanks")
        
        uploaded_images = st.file_uploader(
            "Upload multiple images for this celebrity",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True
        )
        
        if st.button("👤 Add Celebrity") and celebrity_name and uploaded_images:
            with st.spinner(f"Adding {celebrity_name}..."):
                try:
                    # Save uploaded images temporarily
                    temp_paths = []
                    for uploaded_file in uploaded_images:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            temp_paths.append(tmp_file.name)
                    
                    success = recognizer.add_single_celebrity(celebrity_name, temp_paths)
                    
                    # Clean up temporary files
                    for temp_path in temp_paths:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                    
                    if success:
                        st.success(f"✅ Successfully added {celebrity_name}!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Failed to add celebrity!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with tab5:
        st.markdown("**Remove a celebrity from the model**")
        
        if model_info['trained'] and model_info['celebrities']:
            celebrity_to_remove = st.selectbox(
                "Select celebrity to remove:",
                model_info['celebrities']
            )
            
            if st.button("🗑️ Remove Celebrity", type="secondary"):
                with st.spinner(f"Removing {celebrity_to_remove}..."):
                    try:
                        success = recognizer.remove_celebrity(celebrity_to_remove)
                        if success:
                            st.success(f"✅ Successfully removed {celebrity_to_remove}!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to remove celebrity!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.info("No celebrities in model to remove")
    
    # Dataset upload (for demo purposes)
    st.subheader("📁 Upload Dataset")
    st.info("Upload a zip file containing celebrity folders with images")
    
    uploaded_zip = st.file_uploader(
        "Upload dataset ZIP file",
        type=['zip'],
        help="ZIP file should contain folders named after celebrities with their images inside"
    )
    
    if uploaded_zip is not None:
        st.info("Dataset upload functionality would be implemented here")
        st.write("This feature would extract the ZIP and train the model")

def show_model_info_page(recognizer):
    """Display the model information page"""
    st.header("📊 Model Information")
    
    model_info = recognizer.get_model_info()
    
    if model_info['trained']:
        # Model metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Celebrities", model_info['num_celebrities'])
        
        with col2:
            st.metric("Embeddings", model_info['total_embeddings'])
        
        with col3:
            avg_embeddings = model_info['total_embeddings'] / model_info['num_celebrities']
            st.metric("Avg per Celebrity", f"{avg_embeddings:.1f}")
        
        # Celebrity list
        st.subheader("👥 Recognized Celebrities")
        
        if model_info['celebrities']:
            # Create a nice grid layout for celebrities
            cols = st.columns(4)
            for i, celeb in enumerate(model_info['celebrities']):
                with cols[i % 4]:
                    st.markdown(f"**{i+1}.** {celeb}")
        else:
            st.info("No celebrities in the model yet")
        
        # Model actions
        st.subheader("🔧 Model Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Model"):
                if recognizer.save_model():
                    st.success("✅ Model saved successfully!")
                else:
                    st.error("❌ Failed to save model")
        
        with col2:
            if st.button("🔄 Reload Model"):
                if recognizer.load_model():
                    st.success("✅ Model reloaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to reload model")
    
    else:
        st.warning("⚠️ No trained model found")
        st.info("Please train a model using the Model Training page")

def show_settings_page(recognizer):
    """Display the settings page"""
    st.header("⚙️ Settings")
    
    st.subheader("🔧 Model Configuration")
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Minimum confidence level for face recognition"
    )
    
    # Model directory
    model_dir = st.text_input(
        "Model Directory",
        value=recognizer.model_dir,
        help="Directory where model files are stored"
    )
    
    # Advanced settings
    with st.expander("🔬 Advanced Settings"):
        st.checkbox("Enable debug mode", value=False)
        st.checkbox("Save intermediate results", value=True)
        st.checkbox("Auto-save model", value=True)
    
    # System info
    st.subheader("💻 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Python Version:**", sys.version)
        st.write("**OpenCV Version:**", cv2.__version__)
    
    with col2:
        st.write("**NumPy Version:**", np.__version__)
        st.write("**Streamlit Version:**", st.__version__)

if __name__ == "__main__":

    main() 
