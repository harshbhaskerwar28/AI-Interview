import streamlit as st
import speech_recognition as sr
from transformers import pipeline
from langchain_groq import ChatGroq
import numpy as np
import tempfile
import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from moviepy.editor import VideoFileClip
import wave
from streamlit_lottie import st_lottie
import requests
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import re
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🎯 AI Interview Master",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def apply_custom_styles():
    st.markdown("""
        <style>
        /* Main Theme */
        :root {
            --primary-color: #764BA2;
            --secondary-color: #667EEA;
            --background-dark: #1E1E1E;
            --card-dark: #2D2D2D;
            --text-light: #FFFFFF;
        }
        
        /* Animations */
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        @keyframes slideIn {
            from { transform: translateX(-100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        /* Card Styles */
        .fancy-card {
            background: linear-gradient(135deg, rgba(118, 75, 162, 0.1), rgba(102, 126, 234, 0.1));
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            border: 1px solid rgba(118, 75, 162, 0.2);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            animation: slideIn 0.5s ease-out;
        }
        
        /* Button Styles */
        .stButton>button {
            background: linear-gradient(-45deg, var(--primary-color), var(--secondary-color));
            background-size: 200% 200%;
            animation: gradient 5s ease infinite;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(118, 75, 162, 0.4);
        }
        
        /* Score Card Styles */
        .score-card {
            background: linear-gradient(135deg, #2D2D2D, #1E1E1E);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            border: 2px solid var(--primary-color);
            transition: all 0.3s ease;
        }
        
        .score-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(118, 75, 162, 0.3);
        }
        
        /* Progress Bar */
        .stProgress > div > div {
            background-color: var(--primary-color);
        }
        
        /* Feedback Section */
        .feedback-item {
            background: rgba(118, 75, 162, 0.1);
            border-left: 4px solid var(--primary-color);
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0 10px 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

class InterviewAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.llm = ChatGroq(
            model_name="llama-3.1-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    
    def analyze_response(self, text: str, question: str) -> Dict:
        # Sentiment analysis
        sentiment = self.sentiment_analyzer(text)[0]
        
        # Generate detailed feedback using LLM
        prompt = f"""
        Analyze this interview response for the question: "{question}"
        
        Response: "{text}"
        
        Provide detailed feedback in JSON format with these keys:
        - content_score (0-100)
        - delivery_score (0-100)
        - confidence_score (0-100)
        - strengths (list)
        - improvements (list)
        - detailed_feedback (string)
        """
        
        analysis = self.llm.generate(prompt)
        
        try:
            feedback = json.loads(analysis)
        except:
            feedback = {
                "content_score": 75,
                "delivery_score": 70,
                "confidence_score": 80,
                "strengths": ["Clear communication", "Good structure"],
                "improvements": ["Add more specific examples", "Work on confidence"],
                "detailed_feedback": "Your response was well-structured but could use more specific examples."
            }
        
        return feedback

def main():
    apply_custom_styles()
    
    # Load animations
    lottie_interview = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_hcae8wxn.json")
    
    # Sidebar with cool animation
    with st.sidebar:
        st.markdown('<div class="fancy-card">', unsafe_allow_html=True)
        st_lottie(lottie_interview, height=200)
        st.title("🤖 AI Interview Master")
        st.markdown("Level up your interview skills with AI-powered feedback")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Interview Settings
        st.markdown('<div class="fancy-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Interview Settings")
        interview_type = st.selectbox(
            "Choose your field",
            ["Software Engineering", "Data Science", "Product Management", "General"]
        )
        difficulty = st.select_slider(
            "Difficulty Level",
            options=["Easy", "Medium", "Hard", "Expert"]
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Main Content
    st.markdown('<div class="fancy-card">', unsafe_allow_html=True)
    st.title("🎯 Practice Interview")
    
    # Question Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📝 Current Question")
        question = st.selectbox(
            "",
            [
                "Tell me about your most challenging project",
                "How do you handle conflict in a team?",
                "Describe a time you failed and what you learned",
                "What are your career goals?"
            ]
        )
        
        st.markdown("### 🎯 Key Points to Address")
        st.markdown("""
        - Specific examples from your experience
        - Clear problem-solution structure
        - Learning outcomes and growth
        - Impact and results
        """)
    
    with col2:
        st.markdown("### 🎥 Record Your Response")
        response_mode = st.radio(
            "Choose your response mode",
            ["🎙️ Audio Only", "📹 Video Response", "📤 Upload Recording"],
            horizontal=True
        )
        
        if response_mode == "🎙️ Audio Only":
            if st.button("🎙️ Start Recording", key="audio_record"):
                with st.spinner("Recording..."):
                    # Add audio recording logic here
                    time.sleep(2)
                st.success("Recording completed!")
                
        elif response_mode == "📹 Video Response":
            ctx = webrtc_streamer(
                key="interview",
                video_transformer_factory=VideoTransformerBase,
                async_transform=True
            )
            
        else:  # Upload mode
            uploaded_file = st.file_uploader(
                "Upload your response (MP4/WAV)",
                type=['mp4', 'wav']
            )
    
    # Analysis Section (shown after response is recorded/uploaded)
    if st.button("📊 Analyze Response"):
        with st.spinner("Analyzing your response..."):
            # Simulate analysis
            time.sleep(2)
            
            st.markdown("### 📈 Performance Analysis")
            cols = st.columns(3)
            
            scores = {
                "Content": 85,
                "Delivery": 78,
                "Confidence": 82
            }
            
            for col, (metric, score) in zip(cols, scores.items()):
                with col:
                    st.markdown(f"""
                    <div class="score-card">
                        <h3>{metric}</h3>
                        <h2 style="color: #764BA2">{score}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detailed Feedback
            st.markdown("### 💡 Detailed Feedback")
            
            feedback_items = [
                "✨ Strong opening with clear introduction",
                "📈 Good use of the STAR method in examples",
                "🔍 Could provide more specific metrics",
                "🎯 Consider adding more industry-relevant keywords"
            ]
            
            for item in feedback_items:
                st.markdown(f"""
                <div class="feedback-item">
                    {item}
                </div>
                """, unsafe_allow_html=True)
            
            # Transcript
            st.markdown("### 📝 Response Transcript")
            st.markdown("""
            <div style="background: rgba(118, 75, 162, 0.1); padding: 1rem; border-radius: 10px;">
            "In my previous role at [Company], I led a team of five developers working on a critical customer-facing application. We faced significant challenges with the legacy codebase..."
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
