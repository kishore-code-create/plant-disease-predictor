"""
Advanced Plant Leaf Disease Prediction Application
==================================================
A comprehensive Streamlit application for predicting plant diseases from leaf images
with extensive features including disease information, treatment recommendations,
prediction history, and advanced analytics.

Author: Plant Disease Detection System
Version: 2.0
Date: 2024
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
from datetime import datetime
import json
import base64
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import time

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Model configuration
MODEL_PATH = "plant_disease_model_15_class.h5"
IMAGE_SIZE = (150, 150)
CONFIDENCE_THRESHOLD = 0.5

# Class names for the 15 disease categories
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# ============================================================================
# DISEASE INFORMATION DATABASE
# ============================================================================

DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "name": "Bacterial Spot on Bell Pepper",
        "severity": "High",
        "description": "Bacterial spot is a serious disease affecting pepper plants, causing significant yield loss.",
        "symptoms": [
            "Small, dark brown spots with yellow halos on leaves",
            "Raised lesions on fruits",
            "Leaf drop in severe cases",
            "Stunted plant growth"
        ],
        "causes": [
            "Caused by Xanthomonas bacteria",
            "Spreads through contaminated seeds",
            "High humidity and warm temperatures favor disease",
            "Water splash disperses bacteria"
        ],
        "treatment": [
            "Remove and destroy infected plants",
            "Apply copper-based bactericides",
            "Use resistant varieties when available",
            "Improve air circulation around plants",
            "Avoid overhead irrigation"
        ],
        "prevention": [
            "Use certified disease-free seeds",
            "Practice crop rotation (3-4 years)",
            "Maintain proper plant spacing",
            "Disinfect tools regularly",
            "Avoid working with wet plants"
        ]
    },
    "Pepper__bell___healthy": {
        "name": "Healthy Bell Pepper",
        "severity": "None",
        "description": "The plant shows no signs of disease and appears healthy.",
        "symptoms": [
            "Vibrant green leaves",
            "No spots or discoloration",
            "Normal growth pattern",
            "Healthy fruit development"
        ],
        "causes": [
            "Good agricultural practices",
            "Proper nutrition and watering",
            "Disease-free environment"
        ],
        "treatment": [
            "No treatment needed",
            "Continue regular maintenance"
        ],
        "prevention": [
            "Maintain current care practices",
            "Regular monitoring for early detection",
            "Proper fertilization schedule",
            "Adequate watering"
        ]
    },
    "Potato___Early_blight": {
        "name": "Early Blight on Potato",
        "severity": "Medium",
        "description": "Early blight is a common fungal disease affecting potato plants, reducing yield and quality.",
        "symptoms": [
            "Concentric ring patterns on older leaves",
            "Dark brown lesions",
            "Yellowing around spots",
            "Premature leaf drop",
            "Stem lesions in severe cases"
        ],
        "causes": [
            "Caused by Alternaria solani fungus",
            "Warm, humid conditions promote disease",
            "Spreads through infected plant debris",
            "Water splash disperses spores"
        ],
        "treatment": [
            "Apply fungicides (chlorothalonil, mancozeb)",
            "Remove infected lower leaves",
            "Improve air circulation",
            "Use drip irrigation instead of overhead",
            "Apply mulch to prevent soil splash"
        ],
        "prevention": [
            "Use resistant varieties",
            "Crop rotation (3-4 years)",
            "Remove plant debris after harvest",
            "Proper plant spacing",
            "Avoid nitrogen excess"
        ]
    },
    "Potato___Late_blight": {
        "name": "Late Blight on Potato",
        "severity": "Very High",
        "description": "Late blight is a devastating disease that can destroy entire potato crops rapidly.",
        "symptoms": [
            "Water-soaked spots on leaves",
            "White fuzzy growth on undersides",
            "Dark brown lesions on stems",
            "Rapid plant collapse",
            "Brown rot in tubers"
        ],
        "causes": [
            "Caused by Phytophthora infestans",
            "Cool, wet weather ideal for disease",
            "Highly contagious and spreads rapidly",
            "Can survive in infected tubers"
        ],
        "treatment": [
            "Immediate fungicide application",
            "Remove and destroy infected plants",
            "Apply protective fungicides preventatively",
            "Monitor weather conditions closely",
            "Harvest early if disease detected"
        ],
        "prevention": [
            "Use certified disease-free seed potatoes",
            "Plant resistant varieties",
            "Ensure good drainage",
            "Monitor regularly during susceptible periods",
            "Destroy volunteer plants"
        ]
    },
    "Potato___healthy": {
        "name": "Healthy Potato",
        "severity": "None",
        "description": "The potato plant is healthy with no visible disease symptoms.",
        "symptoms": [
            "Dark green, vigorous foliage",
            "No lesions or spots",
            "Normal flowering",
            "Healthy tuber development"
        ],
        "causes": [
            "Optimal growing conditions",
            "Proper cultural practices",
            "Disease prevention measures effective"
        ],
        "treatment": [
            "No treatment required",
            "Continue monitoring"
        ],
        "prevention": [
            "Maintain current practices",
            "Regular scouting",
            "Proper nutrition",
            "Adequate moisture"
        ]
    },
    "Tomato_Bacterial_spot": {
        "name": "Bacterial Spot on Tomato",
        "severity": "High",
        "description": "Bacterial spot causes significant damage to tomato plants and fruit quality.",
        "symptoms": [
            "Small dark spots on leaves",
            "Yellow halos around lesions",
            "Fruit spots with raised white centers",
            "Defoliation in severe cases"
        ],
        "causes": [
            "Xanthomonas bacteria",
            "Warm, wet conditions",
            "Contaminated seeds or transplants",
            "Spreads through water splash"
        ],
        "treatment": [
            "Copper-based sprays",
            "Remove infected tissue",
            "Improve ventilation",
            "Reduce leaf wetness",
            "Use biological controls"
        ],
        "prevention": [
            "Hot water seed treatment",
            "Use disease-free transplants",
            "Avoid overhead watering",
            "Crop rotation",
            "Resistant varieties"
        ]
    },
    "Tomato_Early_blight": {
        "name": "Early Blight on Tomato",
        "severity": "Medium",
        "description": "Early blight commonly affects older tomato leaves and can reduce yield.",
        "symptoms": [
            "Target-like spots on leaves",
            "Brown concentric rings",
            "Leaf yellowing and drop",
            "Stem cankers",
            "Fruit rot near stem"
        ],
        "causes": [
            "Alternaria solani fungus",
            "High humidity",
            "Long periods of leaf wetness",
            "Stressed plants more susceptible"
        ],
        "treatment": [
            "Fungicide applications",
            "Remove lower infected leaves",
            "Mulch to prevent soil splash",
            "Stake plants for air flow",
            "Reduce watering frequency"
        ],
        "prevention": [
            "Resistant varieties",
            "3-year crop rotation",
            "Proper spacing",
            "Remove debris",
            "Balanced fertilization"
        ]
    },
    "Tomato_Late_blight": {
        "name": "Late Blight on Tomato",
        "severity": "Very High",
        "description": "Late blight can devastate tomato crops in days under favorable conditions.",
        "symptoms": [
            "Large brown blotches on leaves",
            "White mold on undersides",
            "Dark lesions on stems",
            "Firm brown spots on fruit",
            "Rapid plant death"
        ],
        "causes": [
            "Phytophthora infestans",
            "Cool, wet weather",
            "High humidity",
            "Spreads very rapidly"
        ],
        "treatment": [
            "Immediate fungicide treatment",
            "Destroy infected plants",
            "Remove infected fruit",
            "Apply preventative sprays",
            "Increase air circulation"
        ],
        "prevention": [
            "Resistant varieties",
            "Greenhouse growing",
            "Rain protection",
            "Regular monitoring",
            "Preventative fungicide program"
        ]
    },
    "Tomato_Leaf_Mold": {
        "name": "Leaf Mold on Tomato",
        "severity": "Medium",
        "description": "Leaf mold primarily affects greenhouse-grown tomatoes with poor ventilation.",
        "symptoms": [
            "Yellow spots on upper leaf surface",
            "Olive-green to brown mold below",
            "Leaves curl and wither",
            "Reduced photosynthesis",
            "Premature fruit ripening"
        ],
        "causes": [
            "Passalora fulva fungus",
            "High humidity (>85%)",
            "Poor air circulation",
            "Temperatures 72-77°F optimal"
        ],
        "treatment": [
            "Improve ventilation",
            "Reduce humidity",
            "Remove infected leaves",
            "Fungicide application",
            "Increase plant spacing"
        ],
        "prevention": [
            "Resistant varieties",
            "Proper greenhouse ventilation",
            "Humidity control",
            "Adequate spacing",
            "Avoid overhead irrigation"
        ]
    },
    "Tomato_Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot on Tomato",
        "severity": "Medium",
        "description": "Septoria leaf spot is a common foliar disease affecting tomato plants.",
        "symptoms": [
            "Small circular spots with dark borders",
            "Gray centers with tiny black dots",
            "Starts on lower leaves",
            "Leaves turn yellow and drop",
            "Can defoliate entire plant"
        ],
        "causes": [
            "Septoria lycopersici fungus",
            "Warm, wet conditions",
            "Spreads through water splash",
            "Survives in plant debris"
        ],
        "treatment": [
            "Fungicide program",
            "Remove infected leaves",
            "Mulch around plants",
            "Cage or stake plants",
            "Avoid wetting foliage"
        ],
        "prevention": [
            "Crop rotation",
            "Remove plant debris",
            "Proper spacing",
            "Drip irrigation",
            "Morning watering"
        ]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Two-Spotted Spider Mite on Tomato",
        "severity": "High",
        "description": "Spider mites are tiny pests that cause significant damage by feeding on plant sap.",
        "symptoms": [
            "Stippling or bronzing of leaves",
            "Fine webbing on plants",
            "Yellow or brown leaves",
            "Leaf drop",
            "Stunted growth"
        ],
        "causes": [
            "Tetranychus urticae",
            "Hot, dry conditions",
            "Dusty environments",
            "Rapid reproduction in warm weather"
        ],
        "treatment": [
            "Miticides or insecticidal soap",
            "Spray water to dislodge mites",
            "Introduce predatory mites",
            "Neem oil application",
            "Remove heavily infested leaves"
        ],
        "prevention": [
            "Maintain humidity",
            "Regular water spray",
            "Avoid water stress",
            "Beneficial insects",
            "Clean growing area"
        ]
    },
    "Tomato__Target_Spot": {
        "name": "Target Spot on Tomato",
        "severity": "Medium",
        "description": "Target spot causes leaf damage and can affect fruit quality in tomatoes.",
        "symptoms": [
            "Brown spots with concentric rings",
            "Affects leaves, stems, and fruit",
            "Gray center with dark border",
            "Leaf yellowing and drop",
            "Reduced yield"
        ],
        "causes": [
            "Corynespora cassiicola fungus",
            "Warm, humid conditions",
            "Poor air circulation",
            "Extended leaf wetness"
        ],
        "treatment": [
            "Fungicide applications",
            "Remove infected plant parts",
            "Improve air flow",
            "Reduce humidity",
            "Proper plant support"
        ],
        "prevention": [
            "Resistant varieties",
            "Adequate spacing",
            "Crop rotation",
            "Remove debris",
            "Drip irrigation"
        ]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "severity": "Very High",
        "description": "A devastating viral disease transmitted by whiteflies causing severe yield loss.",
        "symptoms": [
            "Upward curling of leaves",
            "Yellowing of leaf margins",
            "Stunted plant growth",
            "Reduced fruit set",
            "Small, misshapen fruit"
        ],
        "causes": [
            "Begomovirus transmitted by whiteflies",
            "Warm climate disease",
            "Cannot be cured once infected",
            "Whitefly populations explode in heat"
        ],
        "treatment": [
            "No cure - remove infected plants",
            "Control whitefly vectors",
            "Insecticide for whiteflies",
            "Yellow sticky traps",
            "Reflective mulches"
        ],
        "prevention": [
            "Resistant varieties essential",
            "Insect-proof screens",
            "Control weeds (virus reservoir)",
            "Early planting before whitefly peak",
            "Remove infected plants immediately"
        ]
    },
    "Tomato__Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "severity": "High",
        "description": "A highly contagious viral disease causing mottling and distortion of tomato plants.",
        "symptoms": [
            "Mottled light and dark green leaves",
            "Leaf distortion and curling",
            "Stunted growth",
            "Yellow streaking on fruit",
            "Reduced yield and quality"
        ],
        "causes": [
            "Tobamovirus - very stable",
            "Mechanical transmission",
            "Contaminated tools and hands",
            "Survives in plant debris and seeds"
        ],
        "treatment": [
            "No cure available",
            "Remove infected plants",
            "Sanitize all tools",
            "Wash hands thoroughly",
            "Avoid tobacco products near plants"
        ],
        "prevention": [
            "Resistant varieties",
            "Use certified virus-free seeds",
            "Sanitize tools with bleach",
            "Don't smoke near plants",
            "Control aphids"
        ]
    },
    "Tomato_healthy": {
        "name": "Healthy Tomato",
        "severity": "None",
        "description": "The tomato plant shows excellent health with no disease symptoms.",
        "symptoms": [
            "Deep green foliage",
            "Vigorous growth",
            "No lesions or discoloration",
            "Normal fruit development",
            "Strong stems"
        ],
        "causes": [
            "Optimal growing conditions",
            "Good cultural practices",
            "Effective disease management"
        ],
        "treatment": [
            "No treatment needed",
            "Maintain current care"
        ],
        "prevention": [
            "Continue current practices",
            "Regular monitoring",
            "Balanced nutrition",
            "Proper watering schedule"
        ]
    }
}

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'total_predictions' not in st.session_state:
        st.session_state.total_predictions = 0
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = CONFIDENCE_THRESHOLD
    if 'show_preprocessing' not in st.session_state:
        st.session_state.show_preprocessing = False

# ============================================================================
# MODEL LOADING AND CACHING
# ============================================================================

@st.cache_resource
def load_model():
    """
    Load the trained Keras model with caching for efficiency.
    
    Returns:
        tensorflow.keras.Model: Loaded model
    """
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.session_state.model_loaded = True
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.session_state.model_loaded = False
        return None

# ============================================================================
# IMAGE PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_image(image, show_steps=False):
    """
    Preprocess the uploaded image for model prediction.
    
    Args:
        image: PIL Image object
        show_steps: Boolean to display preprocessing steps
        
    Returns:
        numpy.ndarray: Preprocessed image ready for prediction
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV processing
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    if show_steps:
        st.write("**Step 1:** Original image converted to array")
        st.image(image, caption="Original Image", width=200)
    
    # Resize to model input size
    img_resized = cv2.resize(img_bgr, IMAGE_SIZE)
    
    if show_steps:
        st.write(f"**Step 2:** Resized to {IMAGE_SIZE}")
        st.image(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), 
                caption=f"Resized to {IMAGE_SIZE}", width=200)
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    if show_steps:
        st.write("**Step 3:** Normalized to [0, 1] range")
        st.write(f"Pixel value range: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    if show_steps:
        st.write(f"**Step 4:** Added batch dimension: {img_batch.shape}")
    
    return img_batch

def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    """
    Apply enhancement filters to the image.
    
    Args:
        image: PIL Image object
        brightness: Brightness factor
        contrast: Contrast factor
        sharpness: Sharpness factor
        
    Returns:
        PIL.Image: Enhanced image
    """
    # Brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    # Contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    # Sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)
    
    return image

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def make_prediction(model, preprocessed_image):
    """
    Make prediction using the loaded model.
    
    Args:
        model: Loaded Keras model
        preprocessed_image: Preprocessed image array
        
    Returns:
        numpy.ndarray: Prediction probabilities for all classes
    """
    try:
        predictions = model.predict(preprocessed_image, verbose=0)
        return predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def get_top_predictions(predictions, top_k=3):
    """
    Get top K predictions with class names and confidence scores.
    
    Args:
        predictions: Array of prediction probabilities
        top_k: Number of top predictions to return
        
    Returns:
        list: List of tuples (class_name, confidence)
    """
    top_indices = predictions.argsort()[-top_k:][::-1]
    top_predictions = [
        (CLASS_NAMES[idx], float(predictions[idx]))
        for idx in top_indices
    ]
    return top_predictions

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confidence_bars(predictions_list):
    """
    Create a horizontal bar chart for prediction confidences.
    
    Args:
        predictions_list: List of tuples (class_name, confidence)
        
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    classes = [pred[0].replace('_', ' ') for pred in predictions_list]
    confidences = [pred[1] * 100 for pred in predictions_list]
    
    colors = ['#2ecc71' if conf > 70 else '#f39c12' if conf > 50 else '#e74c3c' 
              for conf in confidences]
    
    fig = go.Figure(data=[
        go.Bar(
            x=confidences,
            y=classes,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{conf:.2f}%' for conf in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence Scores",
        xaxis_title="Confidence (%)",
        yaxis_title="Disease Class",
        height=300,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

def plot_all_predictions(predictions):
    """
    Create a comprehensive bar chart showing all class predictions.
    
    Args:
        predictions: Array of all prediction probabilities
        
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    classes = [name.replace('_', ' ') for name in CLASS_NAMES]
    confidences = predictions * 100
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=confidences,
            marker=dict(
                color=confidences,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Confidence %")
            ),
            text=[f'{conf:.2f}%' for conf in confidences],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title="All Class Predictions",
        xaxis_title="Disease Class",
        yaxis_title="Confidence (%)",
        height=500,
        showlegend=False,
        xaxis={'tickangle': -45},
        margin=dict(l=10, r=10, t=40, b=150)
    )
    
    return fig

def plot_prediction_distribution(history):
    """
    Create a pie chart showing distribution of predicted classes in history.
    
    Args:
        history: List of prediction history dictionaries
        
    Returns:
        plotly.graph_objects.Figure: Pie chart figure
    """
    if not history:
        return None
    
    class_counts = defaultdict(int)
    for record in history:
        class_counts[record['prediction']] += 1
    
    labels = [name.replace('_', ' ') for name in class_counts.keys()]
    values = list(class_counts.values())
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Set3)
        )
    ])
    
    fig.update_layout(
        title="Prediction Distribution",
        height=400,
        showlegend=True
    )
    
    return fig

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_class_name(class_name):
    """
    Format class name for better readability.
    
    Args:
        class_name: Original class name with underscores
        
    Returns:
        str: Formatted class name
    """
    # Replace underscores with spaces
    formatted = class_name.replace('_', ' ')
    # Capitalize each word
    formatted = ' '.join(word.capitalize() for word in formatted.split())
    return formatted

def get_severity_color(severity):
    """
    Get color code based on disease severity.
    
    Args:
        severity: Severity level string
        
    Returns:
        str: HTML color code
    """
    severity_colors = {
        "None": "#2ecc71",      # Green
        "Low": "#3498db",        # Blue
        "Medium": "#f39c12",     # Orange
        "High": "#e67e22",       # Dark Orange
        "Very High": "#e74c3c"   # Red
    }
    return severity_colors.get(severity, "#95a5a6")

def get_severity_emoji(severity):
    """
    Get emoji based on disease severity.
    
    Args:
        severity: Severity level string
        
    Returns:
        str: Emoji character
    """
    severity_emojis = {
        "None": "✅",
        "Low": "ℹ️",
        "Medium": "⚠️",
        "High": "🔶",
        "Very High": "🔴"
    }
    return severity_emojis.get(severity, "❓")

def save_prediction_to_history(image, prediction, confidence, timestamp):
    """
    Save prediction to session history.
    
    Args:
        image: PIL Image object
        prediction: Predicted class name
        confidence: Confidence score
        timestamp: Timestamp of prediction
    """
    # Convert image to base64 for storage
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    record = {
        'timestamp': timestamp,
        'prediction': prediction,
        'confidence': confidence,
        'image': img_str
    }
    
    st.session_state.prediction_history.append(record)
    st.session_state.total_predictions += 1

def export_history_to_csv():
    """
    Export prediction history to CSV format.
    
    Returns:
        pandas.DataFrame: History as DataFrame
    """
    if not st.session_state.prediction_history:
        return None
    
    data = []
    for record in st.session_state.prediction_history:
        data.append({
            'Timestamp': record['timestamp'],
            'Prediction': format_class_name(record['prediction']),
            'Confidence': f"{record['confidence'] * 100:.2f}%"
        })
    
    return pd.DataFrame(data)

# ============================================================================
# UI COMPONENT FUNCTIONS
# ============================================================================

def display_disease_info(class_name):
    """
    Display comprehensive disease information in an organized format.
    
    Args:
        class_name: Name of the predicted disease class
    """
    if class_name not in DISEASE_INFO:
        st.warning("Detailed information not available for this class.")
        return
    
    info = DISEASE_INFO[class_name]
    
    # Disease header with severity
    severity_color = get_severity_color(info['severity'])
    severity_emoji = get_severity_emoji(info['severity'])
    
    st.markdown(f"""
    <div style='background-color: {severity_color}20; padding: 15px; border-radius: 10px; 
                border-left: 5px solid {severity_color}; margin-bottom: 20px;'>
        <h2 style='margin: 0; color: {severity_color};'>
            {severity_emoji} {info['name']}
        </h2>
        <p style='margin: 10px 0 0 0; font-size: 14px;'>
            <strong>Severity Level:</strong> {info['severity']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Description
    st.markdown(f"**📋 Description:**")
    st.write(info['description'])
    st.markdown("---")
    
    # Symptoms
    with st.expander("🔍 **Symptoms** - Click to expand", expanded=True):
        for symptom in info['symptoms']:
            st.markdown(f"- {symptom}")
    
    # Causes
    with st.expander("🦠 **Causes** - Click to expand", expanded=False):
        for cause in info['causes']:
            st.markdown(f"- {cause}")
    
    # Treatment
    with st.expander("💊 **Treatment** - Click to expand", expanded=False):
        for treatment in info['treatment']:
            st.markdown(f"- {treatment}")
    
    # Prevention
    with st.expander("🛡️ **Prevention** - Click to expand", expanded=False):
        for prevention in info['prevention']:
            st.markdown(f"- {prevention}")

def display_prediction_card(rank, class_name, confidence, is_top=False):
    """
    Display a styled prediction card.
    
    Args:
        rank: Rank of prediction (1, 2, 3)
        class_name: Predicted class name
        confidence: Confidence score
        is_top: Whether this is the top prediction
    """
    if is_top:
        border_color = "#2ecc71"
        bg_color = "#2ecc7120"
        icon = "🏆"
    else:
        border_color = "#3498db"
        bg_color = "#3498db10"
        icon = f"#{rank}"
    
    formatted_name = format_class_name(class_name)
    confidence_pct = confidence * 100
    
    st.markdown(f"""
    <div style='background-color: {bg_color}; padding: 15px; border-radius: 10px; 
                border-left: 5px solid {border_color}; margin: 10px 0;'>
        <h3 style='margin: 0; color: {border_color};'>
            {icon} Rank {rank}: {formatted_name}
        </h3>
        <div style='margin-top: 10px;'>
            <div style='background-color: #ecf0f1; border-radius: 10px; height: 30px; 
                        position: relative; overflow: hidden;'>
                <div style='background: linear-gradient(90deg, {border_color}, {border_color}80); 
                            width: {confidence_pct}%; height: 100%; border-radius: 10px;
                            display: flex; align-items: center; justify-content: center;'>
                    <span style='color: white; font-weight: bold; font-size: 16px;'>
                        {confidence_pct:.2f}%
                    </span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_statistics():
    """Display statistics about predictions made in this session."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📊 Total Predictions",
            value=st.session_state.total_predictions
        )
    
    with col2:
        if st.session_state.prediction_history:
            avg_confidence = np.mean([
                record['confidence'] 
                for record in st.session_state.prediction_history
            ]) * 100
            st.metric(
                label="📈 Avg Confidence",
                value=f"{avg_confidence:.2f}%"
            )
        else:
            st.metric(label="📈 Avg Confidence", value="N/A")
    
    with col3:
        if st.session_state.prediction_history:
            disease_count = sum(
                1 for record in st.session_state.prediction_history
                if 'healthy' not in record['prediction'].lower()
            )
            st.metric(
                label="🦠 Diseases Detected",
                value=disease_count
            )
        else:
            st.metric(label="🦠 Diseases Detected", value="0")

def display_history():
    """Display prediction history in a user-friendly format."""
    if not st.session_state.prediction_history:
        st.info("No predictions yet. Upload an image to get started!")
        return
    
    st.subheader(f"📜 Prediction History ({len(st.session_state.prediction_history)} records)")
    
    # Display statistics
    display_statistics()
    
    st.markdown("---")
    
    # Show distribution chart
    fig = plot_prediction_distribution(st.session_state.prediction_history)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Display history table
    df = export_history_to_csv()
    if df is not None:
        st.dataframe(df, use_container_width=True)
        
        # Export button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.prediction_history = []
        st.session_state.total_predictions = 0
        st.rerun()

# ============================================================================
# MAIN APPLICATION PAGES
# ============================================================================

def prediction_page():
    """Main prediction page for uploading and analyzing images."""
    st.title("🌿 Plant Leaf Disease Predictor")
    st.markdown("""
    Upload an image of a plant leaf to detect potential diseases. Our AI model can identify
    **15 different conditions** across peppers, potatoes, and tomatoes with high accuracy.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("❌ Failed to load the prediction model. Please check the model file.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a leaf image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the plant leaf. Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file)
            
            # Image enhancement controls (optional)
            with st.expander("🎨 Image Enhancement (Optional)"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
                with col2:
                    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
                with col3:
                    sharpness = st.slider("Sharpness", 0.5, 2.0, 1.0, 0.1)
                
                if brightness != 1.0 or contrast != 1.0 or sharpness != 1.0:
                    image = enhance_image(image, brightness, contrast, sharpness)
            
            # Display original image
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            # Preprocessing visualization toggle
            show_preprocessing = st.checkbox(
                "Show preprocessing steps",
                value=st.session_state.show_preprocessing,
                help="View how the image is processed before prediction"
            )
            st.session_state.show_preprocessing = show_preprocessing
            
            # Predict button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    preprocessed = preprocess_image(image, show_steps=show_preprocessing)
                    
                    # Make prediction
                    predictions = make_prediction(model, preprocessed)
                    
                    if predictions is not None:
                        # Get top predictions
                        top_predictions = get_top_predictions(predictions, top_k=3)
                        
                        # Save to history
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        save_prediction_to_history(
                            image,
                            top_predictions[0][0],
                            top_predictions[0][1],
                            timestamp
                        )
                        
                        st.success("✅ Analysis complete!")
                        
                        # Display results
                        st.markdown("---")
                        st.header("📊 Prediction Results")
                        
                        # Top prediction
                        st.subheader("🏆 Top Prediction")
                        display_prediction_card(
                            1,
                            top_predictions[0][0],
                            top_predictions[0][1],
                            is_top=True
                        )
                        
                        # Check confidence threshold
                        if top_predictions[0][1] < st.session_state.confidence_threshold:
                            st.warning(
                                f"⚠️ Confidence below threshold "
                                f"({st.session_state.confidence_threshold*100:.0f}%). "
                                f"Results may be unreliable."
                            )
                        
                        # Other top predictions
                        st.subheader("📋 Other Top Predictions")
                        for i, (class_name, confidence) in enumerate(top_predictions[1:], start=2):
                            display_prediction_card(i, class_name, confidence, is_top=False)
                        
                        # Confidence visualization
                        st.markdown("---")
                        st.subheader("📊 Confidence Visualization")
                        fig = plot_confidence_bars(top_predictions)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # All predictions chart
                        with st.expander("📈 View All Class Predictions"):
                            fig_all = plot_all_predictions(predictions)
                            st.plotly_chart(fig_all, use_container_width=True)
                        
                        # Disease information
                        st.markdown("---")
                        st.header("📚 Disease Information")
                        display_disease_info(top_predictions[0][0])
                        
                        # Additional recommendations
                        st.markdown("---")
                        st.info("""
                        **💡 Recommendations:**
                        - For accurate diagnosis, consider uploading multiple images
                        - Consult with agricultural experts for severe cases
                        - Monitor your plants regularly for early detection
                        - Follow recommended treatment and prevention measures
                        """)
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please try uploading a different image or check the file format.")

def batch_prediction_page():
    """Page for batch prediction of multiple images."""
    st.title("📦 Batch Prediction")
    st.markdown("""
    Upload multiple images at once for efficient disease detection across your crop.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("❌ Failed to load the prediction model.")
        return
    
    # Multiple file uploader
    uploaded_files = st.file_uploader(
        "Choose multiple leaf images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload multiple images for batch processing"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} images uploaded")
        
        if st.button("🚀 Process All Images", type="primary", use_container_width=True):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing image {idx + 1}/{len(uploaded_files)}...")
                    
                    # Load and process image
                    image = Image.open(uploaded_file)
                    preprocessed = preprocess_image(image, show_steps=False)
                    predictions = make_prediction(model, preprocessed)
                    
                    if predictions is not None:
                        top_pred = get_top_predictions(predictions, top_k=1)[0]
                        results.append({
                            'filename': uploaded_file.name,
                            'image': image,
                            'prediction': format_class_name(top_pred[0]),
                            'confidence': f"{top_pred[1] * 100:.2f}%",
                            'raw_prediction': top_pred[0],
                            'raw_confidence': top_pred[1]
                        })
                    
                except Exception as e:
                    st.warning(f"⚠️ Error processing {uploaded_file.name}: {str(e)}")
            
            progress_bar.empty()
            status_text.empty()
            
            if results:
                st.success(f"✅ Processed {len(results)} images successfully!")
                
                # Display results
                st.markdown("---")
                st.header("📊 Batch Results")
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Images", len(results))
                with col2:
                    disease_count = sum(
                        1 for r in results 
                        if 'healthy' not in r['raw_prediction'].lower()
                    )
                    st.metric("Diseases Detected", disease_count)
                with col3:
                    avg_conf = np.mean([r['raw_confidence'] for r in results]) * 100
                    st.metric("Avg Confidence", f"{avg_conf:.2f}%")
                
                # Results table
                df = pd.DataFrame([
                    {
                        'Filename': r['filename'],
                        'Prediction': r['prediction'],
                        'Confidence': r['confidence']
                    }
                    for r in results
                ])
                st.dataframe(df, use_container_width=True)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Display individual results
                st.markdown("---")
                st.subheader("🖼️ Individual Results")
                
                cols_per_row = 3
                for i in range(0, len(results), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(results):
                            result = results[i + j]
                            with cols[j]:
                                st.image(result['image'], use_container_width=True)
                                st.caption(f"**{result['filename']}**")
                                st.write(f"**Prediction:** {result['prediction']}")
                                st.write(f"**Confidence:** {result['confidence']}")

def history_page():
    """Page for viewing prediction history."""
    st.title("📜 Prediction History")
    st.markdown("""
    View and manage your prediction history from this session.
    """)
    
    display_history()

def info_page():
    """Information page about diseases and the application."""
    st.title("ℹ️ Information & Guide")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌱 About", 
        "🦠 Disease Guide", 
        "📖 How to Use", 
        "❓ FAQ"
    ])
    
    with tab1:
        st.header("About This Application")
        st.markdown("""
        ### 🌿 Plant Leaf Disease Predictor
        
        This application uses advanced deep learning techniques to identify plant diseases
        from leaf images. Our model is trained to recognize **15 different conditions**
        affecting three major crops:
        
        - **🌶️ Bell Peppers** (2 conditions)
        - **🥔 Potatoes** (3 conditions)
        - **🍅 Tomatoes** (10 conditions)
        
        #### 🎯 Key Features:
        - Real-time disease detection
        - Detailed disease information and treatment recommendations
        - Batch processing capability
        - Prediction history tracking
        - Confidence scoring and visualization
        - Image enhancement tools
        
        #### 🔬 Technology Stack:
        - **Deep Learning Framework:** TensorFlow/Keras
        - **Model Architecture:** Convolutional Neural Network (CNN)
        - **Input Size:** 150x150 pixels
        - **Number of Classes:** 15
        
        #### 📊 Model Performance:
        Our model has been trained on thousands of images and achieves high accuracy
        in identifying plant diseases under various conditions.
        
        #### ⚠️ Important Notes:
        - This tool is for preliminary assessment only
        - Always consult with agricultural experts for final diagnosis
        - Consider environmental factors and plant history
        - Multiple images provide better accuracy
        """)
        
        st.markdown("---")
        st.info("""
        **👨‍💻 Developed by:** Plant Disease Detection System  
        **📅 Version:** 2.0  
        **📧 Contact:** support@plantdisease.ai
        """)
    
    with tab2:
        st.header("🦠 Complete Disease Guide")
        st.markdown("""
        Browse through all the diseases our system can detect. Click on any disease
        to view detailed information about symptoms, causes, treatment, and prevention.
        """)
        
        # Group diseases by plant type
        plant_groups = {
            "Bell Pepper": [],
            "Potato": [],
            "Tomato": []
        }
        
        for class_name in CLASS_NAMES:
            if "Pepper" in class_name:
                plant_groups["Bell Pepper"].append(class_name)
            elif "Potato" in class_name:
                plant_groups["Potato"].append(class_name)
            elif "Tomato" in class_name:
                plant_groups["Tomato"].append(class_name)
        
        for plant, diseases in plant_groups.items():
            st.subheader(f"🌱 {plant}")
            for disease in diseases:
                with st.expander(f"📄 {format_class_name(disease)}"):
                    display_disease_info(disease)
                    st.markdown("---")
    
    with tab3:
        st.header("📖 How to Use This Application")
        
        st.markdown("""
        ### 🚀 Getting Started
        
        #### 1️⃣ Single Image Prediction
        1. Navigate to the **"Predict"** page
        2. Click on the file uploader
        3. Select a clear image of a plant leaf
        4. (Optional) Adjust image enhancement settings
        5. Click **"Analyze Image"**
        6. View results and disease information
        
        #### 2️⃣ Batch Prediction
        1. Go to the **"Batch Prediction"** page
        2. Upload multiple images at once
        3. Click **"Process All Images"**
        4. Download results as CSV
        
        #### 3️⃣ View History
        1. Navigate to the **"History"** page
        2. View all predictions from your session
        3. Download history as CSV
        4. Clear history if needed
        
        ### 📸 Tips for Best Results
        
        #### Image Quality:
        - ✅ Use clear, well-lit images
        - ✅ Capture the entire leaf or affected area
        - ✅ Avoid blurry or low-resolution images
        - ✅ Minimize shadows and glare
        
        #### What to Photograph:
        - ✅ Leaves showing symptoms
        - ✅ Close-up of affected areas
        - ✅ Multiple angles if possible
        - ❌ Avoid heavily damaged or decomposed leaves
        
        #### Lighting:
        - ✅ Natural daylight is best
        - ✅ Even lighting across the leaf
        - ❌ Avoid direct harsh sunlight
        - ❌ Avoid flash photography if possible
        
        ### 🔧 Advanced Features
        
        #### Image Enhancement:
        - Adjust **brightness** for dark images
        - Increase **contrast** to highlight symptoms
        - Apply **sharpness** for better detail
        
        #### Confidence Threshold:
        - Set in sidebar settings
        - Default: 50%
        - Predictions below threshold show warnings
        
        #### Preprocessing Visualization:
        - Enable to see how images are processed
        - Useful for understanding model input
        """)
    
    with tab4:
        st.header("❓ Frequently Asked Questions")
        
        with st.expander("❓ How accurate is the model?"):
            st.markdown("""
            Our model achieves high accuracy on the training dataset. However, accuracy
            in real-world conditions depends on:
            - Image quality
            - Lighting conditions
            - Disease progression stage
            - Similarity to training data
            
            For best results, upload clear, well-lit images and consider multiple angles.
            """)
        
        with st.expander("❓ Can I use this for professional diagnosis?"):
            st.markdown("""
            This tool is designed as a **preliminary assessment aid** and should not replace
            professional agricultural expertise. Always:
            - Consult with agricultural extension officers
            - Consider local disease prevalence
            - Factor in environmental conditions
            - Get laboratory confirmation for critical cases
            """)
        
        with st.expander("❓ What image formats are supported?"):
            st.markdown("""
            Supported formats:
            - JPG / JPEG
            - PNG
            
            Recommended:
            - Resolution: At least 150x150 pixels (higher is better)
            - File size: Under 10MB
            - Format: JPG for best compatibility
            """)
        
        with st.expander("❓ How many images can I process at once?"):
            st.markdown("""
            The batch prediction feature supports multiple images. However:
            - Processing time increases with more images
            - Consider your system's memory
            - Recommended: 10-50 images per batch for optimal performance
            """)
        
        with st.expander("❓ Is my data stored or shared?"):
            st.markdown("""
            Privacy and data security:
            - Images are processed in real-time
            - No images are permanently stored on servers
            - History is session-based only
            - Data resets when you close the browser
            - No personal information is collected
            """)
        
        with st.expander("❓ What if the prediction seems wrong?"):
            st.markdown("""
            If you believe the prediction is incorrect:
            1. Check image quality and lighting
            2. Try uploading from a different angle
            3. Look at the confidence score (low confidence suggests uncertainty)
            4. Review other top predictions
            5. Consult with an agricultural expert
            6. Consider environmental factors not visible in the image
            """)
        
        with st.expander("❓ Can I use this offline?"):
            st.markdown("""
            Currently, this application requires:
            - Internet connection for the web interface
            - Local model file (plant_disease_model_15_class.h5)
            - Python environment with required packages
            
            For offline use, you can run the application locally with the required files.
            """)
        
        with st.expander("❓ What plants are supported?"):
            st.markdown("""
            Currently supported crops:
            - **Bell Peppers:** 2 conditions (bacterial spot, healthy)
            - **Potatoes:** 3 conditions (early blight, late blight, healthy)
            - **Tomatoes:** 10 conditions (various diseases and healthy)
            
            We're working on expanding to more crops in future versions.
            """)

def settings_page():
    """Settings page for application configuration."""
    st.title("⚙️ Settings")
    st.markdown("Configure application preferences and parameters.")
    
    st.markdown("---")
    
    # Confidence threshold setting
    st.subheader("🎯 Prediction Settings")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence_threshold,
        step=0.05,
        help="Predictions below this threshold will show a warning"
    )
    st.session_state.confidence_threshold = confidence
    
    st.info(f"""
    **Current threshold:** {confidence * 100:.0f}%  
    Predictions with confidence below this value will be flagged as potentially unreliable.
    """)
    
    st.markdown("---")
    
    # Display settings
    st.subheader("🎨 Display Settings")
    
    show_preprocessing = st.checkbox(
        "Show preprocessing steps by default",
        value=st.session_state.show_preprocessing
    )
    st.session_state.show_preprocessing = show_preprocessing
    
    st.markdown("---")
    
    # Data management
    st.subheader("📊 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Predictions in History",
            len(st.session_state.prediction_history)
        )
    
    with col2:
        st.metric(
            "Total Predictions",
            st.session_state.total_predictions
        )
    
    if st.button("🗑️ Clear All Data", type="secondary"):
        if st.button("⚠️ Confirm Clear All Data"):
            st.session_state.prediction_history = []
            st.session_state.total_predictions = 0
            st.success("✅ All data cleared!")
            st.rerun()
    
    st.markdown("---")
    
    # About section
    st.subheader("ℹ️ Application Information")
    
    info_cols = st.columns(2)
    
    with info_cols[0]:
        st.markdown("""
        **Model Information:**
        - Model Type: CNN
        - Input Size: 150x150
        - Classes: 15
        - Framework: TensorFlow
        """)
    
    with info_cols[1]:
        st.markdown("""
        **System Status:**
        - Model: ✅ Loaded
        - Version: 2.0
        - Last Updated: 2024
        """)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

def setup_sidebar():
    """Setup sidebar navigation and information."""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/2ecc71/ffffff?text=Plant+Disease", 
                use_container_width=True)
        
        st.markdown("---")
        
        # Navigation
        st.header("📍 Navigation")
        page = st.radio(
            "Go to",
            ["🔍 Predict", "📦 Batch Prediction", "📜 History", "ℹ️ Info", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick stats
        st.header("📊 Quick Stats")
        st.metric("Session Predictions", st.session_state.total_predictions)
        
        if st.session_state.model_loaded:
            st.success("✅ Model Ready")
        else:
            st.error("❌ Model Not Loaded")
        
        st.markdown("---")
        
        # Supported crops
        st.header("🌱 Supported Crops")
        st.markdown("""
        - 🌶️ Bell Peppers (2)
        - 🥔 Potatoes (3)
        - 🍅 Tomatoes (10)
        """)
        
        st.markdown("---")
        
        # Tips
        with st.expander("💡 Quick Tips"):
            st.markdown("""
            - Use clear, well-lit images
            - Capture affected leaf areas
            - Multiple angles help
            - Check confidence scores
            - Consult experts for serious cases
            """)
        
        st.markdown("---")
        st.caption("Plant Disease Predictor v2.0")
        st.caption("© 2024 All Rights Reserved")
    
    return page

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Plant Disease Predictor",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .stApp {
            max-width: 100%;
        }
        .main > div {
            padding-top: 2rem;
        }
        h1 {
            color: #2ecc71;
        }
        h2 {
            color: #27ae60;
        }
        .stButton>button {
            width: 100%;
        }
        div[data-testid="stMetricValue"] {
            font-size: 24px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar and get selected page
    page = setup_sidebar()
    
    # Route to appropriate page
    if page == "🔍 Predict":
        prediction_page()
    elif page == "📦 Batch Prediction":
        batch_prediction_page()
    elif page == "📜 History":
        history_page()
    elif page == "ℹ️ Info":
        info_page()
    elif page == "⚙️ Settings":
        settings_page()

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()