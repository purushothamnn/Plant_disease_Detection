import os
import base64
import requests
import json
import streamlit as st
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Plant Disease Detection System",
    page_icon="🌿",
    layout="centered"
)

# Initialize session state for API key
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# Gemini API URL
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

def encode_image_file(image_file):
    """Encode uploaded image file to base64 string."""
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def analyze_plant_disease(image_data):
    """Send image to Gemini for plant disease analysis."""
    
    if not st.session_state.api_key:
        return {"success": False, "error": "No Gemini API key provided. Please enter your API key in the sidebar."}
    
    # Prepare the prompt for Gemini
    prompt = """
    Analyze this plant image and provide the following information:
    1. Plant identification (species and variety if possible)
    2. Identify any diseases or issues present
    3. Detailed diagnosis of the condition
    4. Treatment recommendations and remedies
    5. Prevention measures
    6. Crop improvement suggestions
    
    Format your response in a structured way with clear headings.
    """
    
    # Prepare the request payload
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_data
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        }
    }
    
    # Send request to Gemini
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": st.session_state.api_key
    }
    
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={st.session_state.api_key}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Process and return the response
        if response.status_code == 200:
            result = response.json()
            try:
                analysis = result["candidates"][0]["content"]["parts"][0]["text"]
                return {"success": True, "analysis": analysis}
            except (KeyError, IndexError) as e:
                return {"success": False, "error": f"Failed to parse Gemini response: {str(e)}"}
        else:
            return {"success": False, "error": f"Gemini API error: {response.status_code} - {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request error: {str(e)}"}

# Add sidebar with information and API key input
with st.sidebar:
    st.header("API Key Setup")
    user_api_key = st.text_input("Enter your Gemini API Key", 
                                value=st.session_state.api_key, 
                                type="password")
    if user_api_key:
        st.session_state.api_key = user_api_key
        
    st.header("About")
    st.info("""
    This app helps farmers and gardeners identify plant diseases and get immediate treatment recommendations.
    
    **Features:**
    - Plant identification
    - Disease detection
    - Treatment recommendations
    - Prevention measures
    - Crop improvement tips
    
    Powered by Google Gemini AI
    """)

# App title and description
st.title("🌿 Plant Disease Detection System")
st.markdown("""
Upload an image of a plant to analyze for diseases and get treatment recommendations.
This system uses Google's Gemini AI to identify plants, detect diseases, and suggest remedies.
""")

# Check if API key is provided
if not st.session_state.api_key:
    st.warning("⚠️ Please enter your Gemini API Key in the sidebar to use this application.")
else:
    # File uploader widget
    uploaded_file = st.file_uploader("Choose a plant image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Plant Image", use_column_width=True)
        
        # Add an analyze button
        if st.button("Analyze Plant"):
            with st.spinner("Analyzing plant image... Please wait."):
                # Convert image to base64
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                image_b64 = base64.b64encode(img_byte_arr).decode('utf-8')
                
                # Call Gemini API
                result = analyze_plant_disease(image_b64)
                
                # Display results
                if result["success"]:
                    st.success("Analysis complete!")
                    st.markdown("## Analysis Results")
                    st.markdown(result["analysis"])
                else:
                    st.error(f"Error: {result['error']}")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and Google Gemini AI*")