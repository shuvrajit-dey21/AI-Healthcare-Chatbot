import streamlit as st
import json
import nltk
import os
import datetime
import speech_recognition as sr
import pandas as pd
import matplotlib.pyplot as plt
from decouple import config, UndefinedValueError
from PIL import Image
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import base64
from io import BytesIO
import requests
import re
import PyPDF2
import docx
import random

# Download necessary NLTK data - setting download_dir explicitly
nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download punkt tokenizer data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

# Download stopwords data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)

# Configuration and Environment Setup
try:
    HUGGINGFACE_API_KEY = config('HUGGINGFACE_API_KEY', default='')
    MODEL_NAME = config('MODEL_NAME', default='google/flan-t5-base')
    MAX_TOKENS = config('MAX_TOKENS', default=150, cast=int)
    TEMPERATURE = config('TEMPERATURE', default=0.7, cast=float)
    APP_TITLE = config('APP_TITLE', default='AI Healthcare Assistant')
    DISCLAIMER = config('DISCLAIMER', default='⚠️ Always consult a healthcare provider for medical advice.')
    ENABLE_ANALYTICS = config('ENABLE_ANALYTICS', default=False, cast=bool)
    ANALYTICS_FILE = config('ANALYTICS_FILE', default='usage_analytics.json')
except UndefinedValueError:
    st.error("Environment variables not set properly. Please check your .env file.")

# Emergency Keywords
EMERGENCY_KEYWORDS = [
    "emergency", "heart attack", "stroke", "bleeding", "suicide", 
    "unconscious", "not breathing", "severe pain", "overdose", "911"
]

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}

if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Analytics functions
def log_interaction(user_input, response, feedback=None):
    if not ENABLE_ANALYTICS:
        return
    
    timestamp = datetime.datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "user_input": user_input,
        "response_length": len(response),
        "feedback": feedback
    }
    
    if os.path.exists(ANALYTICS_FILE):
        with open(ANALYTICS_FILE, 'r') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []
    
    logs.append(log_entry)
    
    with open(ANALYTICS_FILE, 'w') as f:
        json.dump(logs, f, indent=2)

def detect_emergency(text):
    """Check if the user input contains emergency keywords"""
    try:
        # Use word_tokenize if available
        tokens = word_tokenize(text.lower())
    except:
        # Fallback to simple splitting if NLTK tokenizer is not available
        tokens = text.lower().split()
    
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in tokens or keyword in text.lower():
            return True
    return False

def display_emergency_message():
    st.error("""
    🚨 **EMERGENCY DETECTED**
    
    If you are experiencing a medical emergency:
    
    1. Call emergency services immediately (911 in US/Canada)
    2. Do not wait for online advice
    3. Seek immediate in-person medical attention
    
    This AI assistant is not equipped to handle emergencies.
    """)

def format_ai_response(text):
    """Format and clean up the AI-generated response"""
    # Remove any potential system message or prompt remnants
    text = re.sub(r'^Assistant:|^AI:', '', text).strip()
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    
    # Ensure proper formatting for lists
    text = re.sub(r'(?m)^(\d+)\.(?! )', r'\1. ', text)
    text = re.sub(r'(?m)^[•*-](?! )', r'• ', text)
    
    # Add proper line breaks between sections
    text = re.sub(r'([.!?])\s*(\d+\.)', r'\1\n\n\2', text)
    
    # Ensure consistent bullet point formatting
    text = re.sub(r'(?m)^[-*](?! )', '• ', text)
    
    # Add disclaimer if not present
    if not any(phrase in text.lower() for phrase in ['consult', 'healthcare provider', 'medical professional']):
        text += "\n\nRemember: This information is for educational purposes only. Always consult with a healthcare professional for personalized medical advice."
    
    return text.strip()

def analyze_query(query):
    """
    Analyze the user query to extract key information and context.
    Returns a tuple of (main_topic, subtopic, context_details)
    """
    # Normalize the query
    query = query.lower().strip()
    
    # Extract potential temporal context
    temporal_markers = {
        'frequency': r'(how (often|frequently)|daily|weekly|monthly)',
        'duration': r'(how long|duration|for how many|since when)',
        'time_of_day': r'(morning|evening|night|afternoon)',
    }
    
    context = {
        'temporal': {},
        'severity': None,
        'demographic': None,
        'condition_history': None
    }
    
    # Check for temporal context
    for marker_type, pattern in temporal_markers.items():
        if re.search(pattern, query):
            context['temporal'][marker_type] = re.search(pattern, query).group()
    
    # Check for severity indicators
    severity_patterns = r'(severe|mild|moderate|intense|extreme|slight)'
    severity_match = re.search(severity_patterns, query)
    if severity_match:
        context['severity'] = severity_match.group()
    
    # Check for demographic information
    demographic_pattern = r'(child|adult|elderly|senior|year[s]? old|male|female)'
    demographic_match = re.search(demographic_pattern, query)
    if demographic_match:
        context['demographic'] = demographic_match.group()
    
    # Check for condition history
    history_pattern = r'(chronic|recurring|persistent|ongoing|new|sudden|gradual)'
    history_match = re.search(history_pattern, query)
    if history_match:
        context['condition_history'] = history_match.group()
    
    # Identify main topic and subtopic
    main_topic = None
    subtopic = None
    
    # Enhanced topic detection
    topics = {
        'covid': {
            'keywords': ['covid', 'coronavirus', 'sars-cov-2', 'pandemic'],
            'subtopics': {
                'symptoms': ['symptom', 'sign', 'feel'],
                'prevention': ['prevent', 'protect', 'avoid'],
                'treatment': ['treat', 'cure', 'medicine'],
                'testing': ['test', 'positive', 'negative'],
                'vaccination': ['vaccine', 'shot', 'booster']
            }
        },
        'mental_health': {
            'keywords': ['anxiety', 'depression', 'stress', 'mental', 'psychological'],
            'subtopics': {
                'symptoms': ['feeling', 'symptom', 'sign'],
                'coping': ['cope', 'manage', 'deal'],
                'treatment': ['therapy', 'counseling', 'medication'],
                'prevention': ['prevent', 'avoid', 'reduce']
            }
        },
        'nutrition': {
            'keywords': ['diet', 'food', 'nutrition', 'eating', 'meal'],
            'subtopics': {
                'healthy_eating': ['healthy', 'balanced', 'nutritious'],
                'weight_management': ['weight', 'calories', 'loss'],
                'special_diets': ['vegan', 'vegetarian', 'keto', 'paleo'],
                'supplements': ['vitamin', 'mineral', 'supplement']
            }
        },
        'exercise': {
            'keywords': ['exercise', 'workout', 'fitness', 'training'],
            'subtopics': {
                'cardio': ['cardio', 'aerobic', 'running', 'swimming'],
                'strength': ['strength', 'weight', 'muscle'],
                'flexibility': ['stretch', 'yoga', 'flexibility'],
                'planning': ['plan', 'schedule', 'routine']
            }
        },
        'sleep': {
            'keywords': ['sleep', 'insomnia', 'rest', 'nap'],
            'subtopics': {
                'quality': ['quality', 'better', 'improve'],
                'disorders': ['disorder', 'apnea', 'insomnia'],
                'habits': ['habit', 'routine', 'schedule'],
                'environment': ['bedroom', 'environment', 'noise', 'light']
            }
        }
    }
    
    # Find main topic
    for topic, data in topics.items():
        if any(keyword in query for keyword in data['keywords']):
            main_topic = topic
            # Find subtopic
            for sub, keywords in data['subtopics'].items():
                if any(keyword in query for keyword in keywords):
                    subtopic = sub
                    break
            break
    
    return main_topic, subtopic, context

def get_chatbot_response(user_input, uploaded_file=None):
    """Get a response using HuggingFace's inference API or fallback to a rule-based system"""
    try:
        # Prepare the prompt with system message and context
        system_message = """You are an advanced healthcare AI assistant trained to provide detailed, accurate medical information. 
Your responses should be:
1. Comprehensive and well-structured
2. Evidence-based and up-to-date
3. Easy to understand with step-by-step explanations
4. Include relevant medical context and background
5. Always emphasize the importance of consulting healthcare professionals

Remember to:
- Break down complex medical concepts
- Use bullet points and numbered lists for clarity
- Provide specific examples when relevant
- Include preventive measures and lifestyle recommendations
- Always maintain medical accuracy while being accessible

Important: Always emphasize that you're not a replacement for professional medical advice."""
        
        # Build conversation history
        conversation_history = []
        for message in st.session_state.messages[-3:]:  # Include last 3 messages for context
            prefix = "User: " if message["role"] == "user" else "Assistant: "
            conversation_history.append(f"{prefix}{message['content']}")
        
        # Format conversation history as a string
        conversation_str = "\n".join(conversation_history)
        
        # Prepare content
        content = user_input
        
        # If there's an uploaded file, include its context
        if uploaded_file is not None:
            file_content = extract_text_from_file(uploaded_file)
            content += f"\n\nContext from uploaded document: {file_content}"
            
        # Create the full prompt with better structure
        full_prompt = f"""{system_message}

Previous Conversation:
{conversation_str}

User Question: {content}

Please provide a detailed, step-by-step response that includes:
1. Clear explanation of the topic
2. Relevant medical context
3. Practical recommendations
4. Important considerations
5. When to seek professional help"""
        
        # Try using HuggingFace API if API key is provided
        if HUGGINGFACE_API_KEY:
            try:
                st.info("Connecting to HuggingFace API for optimal response...")
                response = requests.post(
                    f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
                    headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
                    json={
                        "inputs": full_prompt,
                        "parameters": {
                            "max_length": MAX_TOKENS,
                            "temperature": TEMPERATURE,
                            "top_p": 0.95,
                            "do_sample": True,
                            "top_k": 50,
                            "repetition_penalty": 1.2,
                            "num_return_sequences": 1,
                            "length_penalty": 1.5
                        }
                    },
                    timeout=30  # Increased timeout for longer responses
                )
                
                # Check if the response is valid
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats based on model type
                    if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                        # Some models return a list of outputs
                        generated_text = result[0]["generated_text"].replace(full_prompt, "").strip()
                        if generated_text:  # Check if we got a meaningful response
                            # Post-process the response to ensure proper formatting
                            formatted_response = format_ai_response(generated_text)
                            return formatted_response
                    elif isinstance(result, dict) and "generated_text" in result:
                        # Some models return a single output
                        generated_text = result["generated_text"].replace(full_prompt, "").strip()
                        if generated_text:  # Check if we got a meaningful response
                            # Post-process the response to ensure proper formatting
                            formatted_response = format_ai_response(generated_text)
                            return formatted_response
                    
                    # If we reach here, we didn't get a useful response from the API
                    st.warning("The model returned an empty or invalid response. Using fallback system...")
                elif response.status_code == 503:
                    # Model is still loading
                    st.warning("The HuggingFace model is still loading. Using fallback response system...")
                else:
                    # Other errors
                    st.warning(f"API Error (Status code: {response.status_code}). Using fallback response system...")
            except requests.exceptions.Timeout:
                st.warning("The request to the HuggingFace model timed out. Using fallback response system...")
            except requests.exceptions.ConnectionError:
                st.warning("Connection error. Check your internet connection. Using fallback response system...")
            except Exception as e:
                st.warning(f"HuggingFace API error: {str(e)}. Using fallback response system...")
        else:
            st.info("No HuggingFace API key provided. Using built-in healthcare knowledge system.")
        
        # Fallback to our enhanced rule-based response system
        return generate_fallback_response(content)
        
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        return "I'm having trouble connecting to my knowledge base right now. Please try again in a moment."

def generate_fallback_response(query):
    """Generate a response using the built-in healthcare knowledge system"""
    main_topic, subtopic, context = analyze_query(query)
    
    # Base responses for different topics and subtopics
    responses = {
        'covid': {
            'symptoms': [
                "Common COVID-19 symptoms include fever, cough, and fatigue. {severity} {history}",
                "COVID-19 typically presents with respiratory symptoms like shortness of breath and loss of taste/smell. {severity} {history}",
                "Initial COVID-19 signs often include fever, dry cough, and tiredness. {severity} {history}"
            ],
            'prevention': [
                "Key COVID-19 prevention measures include vaccination, wearing masks, and maintaining social distance. {temporal}",
                "To prevent COVID-19: wash hands frequently, avoid crowded spaces, and ensure good ventilation. {temporal}",
                "Protect yourself from COVID-19 by staying up to date with vaccines and following local health guidelines. {temporal}"
            ]
        },
        'mental_health': {
            'symptoms': [
                "Common mental health symptoms include changes in mood, sleep patterns, and energy levels. {severity} {history}",
                "Mental health concerns often manifest as anxiety, persistent sadness, or difficulty concentrating. {severity} {history}",
                "Watch for changes in appetite, sleep, and daily functioning as signs of mental health issues. {severity} {history}"
            ],
            'treatment': [
                "Mental health treatment options include therapy, counseling, and sometimes medication. {demographic}",
                "Professional mental health care can involve different approaches like CBT, mindfulness, or group therapy. {demographic}",
                "Treatment plans are personalized and may combine different therapeutic approaches. {demographic}"
            ]
        },
        'nutrition': {
            'healthy_eating': [
                "A balanced diet should include a variety of fruits, vegetables, whole grains, and lean proteins. {temporal}",
                "Focus on eating whole, unprocessed foods and maintaining regular meal times. {temporal}",
                "Consider incorporating different colored fruits and vegetables for a range of nutrients. {temporal}"
            ],
            'weight_management': [
                "Sustainable weight management involves balanced nutrition and regular physical activity. {demographic}",
                "Focus on creating healthy, sustainable habits rather than quick fixes. {demographic}",
                "Consider tracking your food intake and exercise to understand your patterns better. {demographic}"
            ],
            'special_diets': [
                "When following a {context} diet, ensure you're meeting all nutritional requirements. {temporal}",
                "Special diets should be planned carefully to avoid nutrient deficiencies. {temporal}",
                "Consider consulting with a registered dietitian for personalized dietary advice. {temporal}"
            ]
        },
        'exercise': {
            'cardio': [
                "Aim for at least 150 minutes of moderate cardio activity per week. {demographic}",
                "Choose activities you enjoy like walking, swimming, or cycling. {demographic}",
                "Start gradually and increase intensity as your fitness improves. {demographic}"
            ],
            'strength': [
                "Include strength training 2-3 times per week for optimal health. {temporal}",
                "Focus on major muscle groups and proper form during exercises. {temporal}",
                "Allow adequate rest between strength training sessions. {temporal}"
            ],
            'flexibility': [
                "Regular stretching can improve flexibility and reduce injury risk. {severity}",
                "Consider incorporating yoga or other flexibility exercises into your routine. {severity}",
                "Stretch major muscle groups daily, especially after exercise. {severity}"
            ]
        },
        'sleep': {
            'quality': [
                "Good sleep quality involves both duration and consistency. {temporal}",
                "Create a relaxing bedtime routine to improve sleep quality. {temporal}",
                "Aim for 7-9 hours of uninterrupted sleep each night. {temporal}"
            ],
            'disorders': [
                "Sleep disorders can significantly impact overall health. {severity} {history}",
                "Common sleep disorders include insomnia, sleep apnea, and restless leg syndrome. {severity} {history}",
                "Consider a sleep study if you experience persistent sleep issues. {severity} {history}"
            ],
            'habits': [
                "Maintain consistent sleep and wake times, even on weekends. {temporal}",
                "Avoid screens and caffeine close to bedtime. {temporal}",
                "Create a sleep-friendly environment that's dark, quiet, and cool. {temporal}"
            ]
        },
        'diabetes': {
            'management': [
                "Regular blood sugar monitoring is essential for diabetes management. {temporal}",
                "Maintain a consistent meal schedule and balanced diet. {temporal}",
                "Work with your healthcare team to develop an appropriate management plan. {temporal}"
            ],
            'symptoms': [
                "Common diabetes symptoms include increased thirst, frequent urination, and fatigue. {severity} {history}",
                "Watch for signs like slow-healing wounds and blurred vision. {severity} {history}",
                "Monitor for sudden changes in blood sugar levels. {severity} {history}"
            ],
            'prevention': [
                "Maintain a healthy weight and regular physical activity to prevent type 2 diabetes. {demographic}",
                "Choose foods with a low glycemic index and limit processed sugars. {demographic}",
                "Regular health screenings can help detect pre-diabetes early. {demographic}"
            ]
        },
        'heart_health': {
            'prevention': [
                "Maintain healthy blood pressure and cholesterol levels through diet and exercise. {temporal}",
                "Regular cardiovascular exercise supports heart health. {temporal}",
                "Avoid smoking and limit alcohol consumption for heart health. {temporal}"
            ],
            'symptoms': [
                "Watch for signs like chest pain, shortness of breath, or irregular heartbeat. {severity} {history}",
                "Heart attack symptoms can include arm pain and jaw discomfort. {severity} {history}",
                "Some heart conditions may cause fatigue or dizziness. {severity} {history}"
            ],
            'risk_factors': [
                "Key risk factors include high blood pressure, high cholesterol, and smoking. {demographic}",
                "Family history and age can affect heart disease risk. {demographic}",
                "Lifestyle choices significantly impact heart health. {demographic}"
            ]
        }
    }
    
    # Select appropriate response base
    if main_topic in responses and subtopic in responses[main_topic]:
        response_templates = responses[main_topic][subtopic]
        base_response = random.choice(response_templates)
    else:
        # Fallback to general response if topic/subtopic not found
        base_response = "I understand you're asking about {topic}. {general_advice}"
    
    # Format response with context
    severity_text = f"The symptoms you describe are {context['severity']}" if context['severity'] else ""
    history_text = f"Given this is a {context['condition_history']} condition" if context['condition_history'] else ""
    demographic_text = f"For {context['demographic']} patients" if context['demographic'] else ""
    temporal_text = ""
    if context['temporal']:
        temporal_markers = []
        for marker_type, value in context['temporal'].items():
            if value:
                temporal_markers.append(value)
        if temporal_markers:
            temporal_text = f"Regarding {', '.join(temporal_markers)}"
    
    # Format the response
    response = base_response.format(
        severity=severity_text,
        history=history_text,
        demographic=demographic_text,
        temporal=temporal_text,
        topic=main_topic if main_topic else "health",
        general_advice="I recommend consulting with a healthcare professional for personalized advice."
    )
    
    # Add a disclaimer
    response += "\n\nPlease note: This information is for educational purposes only. Always consult with a healthcare professional for medical advice."
    
    return response

def voice_to_text():
    """Convert voice to text using the microphone"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        audio = r.listen(source)
        st.info("Processing your speech...")
        
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand what you said."
    except sr.RequestError:
        return "Sorry, I'm having trouble accessing the speech recognition service."

def get_chat_bubble_css():
    """Return CSS for chat bubbles based on current theme"""
    if st.session_state.theme == "dark":
        return """
        <style>
        .user-bubble {
            background-color: #2e2e2e;
            color: white;
            padding: 10px 15px;
            border-radius: 20px 20px 0 20px;
            margin: 5px 0;
            max-width: 80%;
            margin-left: auto;
            display: inline-block;
        }
        .assistant-bubble {
            background-color: #0078ff;
            color: white;
            padding: 10px 15px;
            border-radius: 20px 20px 20px 0;
            margin: 5px 0;
            max-width: 80%;
            display: inline-block;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            width: 100%;
        }
        .user-container {
            display: flex;
            justify-content: flex-end;
            width: 100%;
        }
        .assistant-container {
            display: flex;
            justify-content: flex-start;
            width: 100%;
        }
        </style>
        """
    else:
        return """
        <style>
        .user-bubble {
            background-color: #e6e6e6;
            color: black;
            padding: 10px 15px;
            border-radius: 20px 20px 0 20px;
            margin: 5px 0;
            max-width: 80%;
            margin-left: auto;
            display: inline-block;
        }
        .assistant-bubble {
            background-color: #0078ff;
            color: white;
            padding: 10px 15px;
            border-radius: 20px 20px 20px 0;
            margin: 5px 0;
            max-width: 80%;
            display: inline-block;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            width: 100%;
        }
        .user-container {
            display: flex;
            justify-content: flex-end;
            width: 100%;
        }
        .assistant-container {
            display: flex;
            justify-content: flex-start;
            width: 100%;
        }
        </style>
        """

def export_chat_history():
    """Export the chat history as text"""
    if not st.session_state.messages:
        return "No chat history to export."
    
    chat_export = ""
    for message in st.session_state.messages:
        if message["role"] == "user":
            chat_export += f"User: {message['content']}\n\n"
        else:
            chat_export += f"Assistant: {message['content']}\n\n"
    
    return chat_export

def get_analytics_data():
    """Get analytics data from the log file"""
    if not os.path.exists(ANALYTICS_FILE):
        return None
    
    try:
        with open(ANALYTICS_FILE, 'r') as f:
            logs = json.load(f)
        
        # Create a DataFrame
        df = pd.DataFrame(logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        return df
    except Exception as e:
        st.error(f"Error processing analytics: {str(e)}")
        return None

def extract_text_from_file(uploaded_file):
    """Extract text from different file types"""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'txt':
            # For text files
            return uploaded_file.getvalue().decode('utf-8')
        
        elif file_type == 'pdf':
            # For PDF files
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + "\n"
            return text
        
        elif file_type == 'docx':
            # For DOCX files
            doc = docx.Document(uploaded_file)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        else:
            return "Unsupported file type. Please upload a txt, pdf, or docx file."
    
    except Exception as e:
        return f"Error extracting text from file: {str(e)}"

def toggle_theme():
    """Toggle between light and dark themes"""
    if st.session_state.theme == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"

def main():
    # Apply theme CSS
    st.markdown(get_chat_bubble_css(), unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Theme toggle
        theme_btn = st.button("Toggle Light/Dark Theme")
        if theme_btn:
            toggle_theme()
        
        st.subheader("Voice Input")
        voice_btn = st.button("🎤 Speak your question")
        
        st.subheader("Upload Health Document")
        uploaded_file = st.file_uploader("Upload a file for context", type=["txt", "pdf", "docx"])
        
        st.subheader("Chat Options")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.feedback_given = {}
            st.success("Chat history cleared!")
        
        export_btn = st.button("📤 Export Chat")
        if export_btn:
            chat_text = export_chat_history()
            # Create a download link
            b64 = base64.b64encode(chat_text.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="chat_export.txt">Download Chat History</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        # API Configuration
        st.subheader("API Settings")
        if not HUGGINGFACE_API_KEY:
            st.warning("⚠️ No HuggingFace API key configured. Using built-in healthcare knowledge system. For more advanced responses, add your API key to the .env file.")
        else:
            st.success("✅ HuggingFace API configured.")
        
        # Analytics section (if enabled)
        if ENABLE_ANALYTICS and st.sidebar.checkbox("Show Analytics"):
            st.subheader("Usage Analytics")
            df = get_analytics_data()
            if df is not None:
                interactions_by_date = df.groupby('date').size()
                
                st.line_chart(interactions_by_date)
                st.write(f"Total interactions: {len(df)}")
    
    # Main app layout
    st.title(APP_TITLE)
    st.markdown(f"*{DISCLAIMER}*")
    
    # Display the chat history
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f'<div class="user-container"><div class="user-bubble">{message["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-container"><div class="assistant-bubble">{message["content"]}</div></div>', unsafe_allow_html=True)
            
            # Add feedback buttons if not yet given for this message
            if i not in st.session_state.feedback_given:
                col1, col2, col3 = st.columns([1, 1, 10])
                with col1:
                    if st.button("👍", key=f"like_{i}"):
                        st.session_state.feedback_given[i] = "positive"
                        log_interaction(
                            st.session_state.messages[i-1]["content"] if i > 0 else "", 
                            message["content"], 
                            "positive"
                        )
                        st.success("Thank you for your feedback!")
                        st.rerun()
                with col2:
                    if st.button("👎", key=f"dislike_{i}"):
                        st.session_state.feedback_given[i] = "negative"
                        log_interaction(
                            st.session_state.messages[i-1]["content"] if i > 0 else "", 
                            message["content"], 
                            "negative"
                        )
                        
                        # Add a report option for negative feedback
                        report_reason = st.text_input(
                            "What was wrong with this response?", 
                            key=f"report_{i}"
                        )
                        if report_reason:
                            st.info("Thank you for helping us improve!")
                            st.session_state.feedback_given[i] = f"negative: {report_reason}"
                            st.rerun()
    
    # Get voice input if button was clicked
    voice_text = None
    if 'voice_btn' in locals() and voice_btn:
        voice_text = voice_to_text()
        if voice_text and voice_text != "Sorry, I couldn't understand what you said." and voice_text != "Sorry, I'm having trouble accessing the speech recognition service.":
            st.info(f"You said: {voice_text}")
    
    # Sample health topics for quick questions
    st.markdown("""
    <style>
    .health-topic-header {
        font-size: 1.5em;
        margin-bottom: 1em;
        color: #0078ff;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="health-topic-header">Common Health Topics</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Create empty container for potential user input from buttons
    if "topic_question" not in st.session_state:
        st.session_state.topic_question = ""
    
    # Create a placeholder for the loading spinner
    loading_placeholder = st.empty()
        
    with col1:
        if st.button("😷 COVID-19"):
            st.session_state.topic_question = "What are the symptoms of COVID-19?"
            # Add the question to chat history and get response immediately
            st.session_state.messages.append({"role": "user", "content": st.session_state.topic_question})
            with loading_placeholder.container():
                st.markdown("""
                <div style="display: flex; align-items: center; gap: 10px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                    <div style="width: 20px; height: 20px; border: 3px solid #0078ff; border-top: 3px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    <span style="color: #0078ff; font-weight: 500;">Generating response...</span>
                </div>
                <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                </style>
                """, unsafe_allow_html=True)
                response = get_chatbot_response(st.session_state.topic_question, None)
            loading_placeholder.empty()
            st.session_state.messages.append({"role": "assistant", "content": response})
            log_interaction(st.session_state.topic_question, response)
            st.rerun()
    with col2:
        if st.button("🤕 Headaches"):
            st.session_state.topic_question = "What causes frequent headaches?"
            st.session_state.messages.append({"role": "user", "content": st.session_state.topic_question})
            with loading_placeholder.container():
                st.markdown("""
                <div style="display: flex; align-items: center; gap: 10px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                    <div style="width: 20px; height: 20px; border: 3px solid #0078ff; border-top: 3px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    <span style="color: #0078ff; font-weight: 500;">Generating response...</span>
                </div>
                """, unsafe_allow_html=True)
                response = get_chatbot_response(st.session_state.topic_question, None)
            loading_placeholder.empty()
            st.session_state.messages.append({"role": "assistant", "content": response})
            log_interaction(st.session_state.topic_question, response)
            st.rerun()
    with col3:
        if st.button("💤 Sleep Issues"):
            st.session_state.topic_question = "How can I improve my sleep quality?"
            st.session_state.messages.append({"role": "user", "content": st.session_state.topic_question})
            with loading_placeholder.container():
                st.markdown("""
                <div style="display: flex; align-items: center; gap: 10px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                    <div style="width: 20px; height: 20px; border: 3px solid #0078ff; border-top: 3px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    <span style="color: #0078ff; font-weight: 500;">Generating response...</span>
                </div>
                """, unsafe_allow_html=True)
                response = get_chatbot_response(st.session_state.topic_question, None)
            loading_placeholder.empty()
            st.session_state.messages.append({"role": "assistant", "content": response})
            log_interaction(st.session_state.topic_question, response)
            st.rerun()
    with col4:
        if st.button("🏃 Exercise"):
            st.session_state.topic_question = "What exercise is best for heart health?"
            st.session_state.messages.append({"role": "user", "content": st.session_state.topic_question})
            with loading_placeholder.container():
                st.markdown("""
                <div style="display: flex; align-items: center; gap: 10px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                    <div style="width: 20px; height: 20px; border: 3px solid #0078ff; border-top: 3px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    <span style="color: #0078ff; font-weight: 500;">Generating response...</span>
                </div>
                """, unsafe_allow_html=True)
                response = get_chatbot_response(st.session_state.topic_question, None)
            loading_placeholder.empty()
            st.session_state.messages.append({"role": "assistant", "content": response})
            log_interaction(st.session_state.topic_question, response)
            st.rerun()
    
    # Text input for user query - prioritize topic questions if selected
    user_input = st.text_input(
        "How can I assist you today?", 
        value=st.session_state.topic_question if st.session_state.topic_question else voice_text if voice_text else ""
    )
    
    # Clear the topic question after it's been used
    if st.session_state.topic_question and user_input == st.session_state.topic_question:
        st.session_state.topic_question = ""
        
    submit_button = st.button("Submit")
    
    if submit_button and user_input:
        # Check for emergency keywords
        if detect_emergency(user_input):
            display_emergency_message()
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get response from the AI model
        with loading_placeholder.container():
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 10px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                <div style="width: 20px; height: 20px; border: 3px solid #0078ff; border-top: 3px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                <span style="color: #0078ff; font-weight: 500;">Generating response...</span>
            </div>
            """, unsafe_allow_html=True)
            response = get_chatbot_response(user_input, uploaded_file)
        loading_placeholder.empty()
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Log the interaction
        log_interaction(user_input, response)
        
        # Rerun to update the UI
        st.rerun()

if __name__ == "__main__":
    main()
