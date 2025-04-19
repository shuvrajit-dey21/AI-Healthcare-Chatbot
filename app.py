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
    # Primary model: Medical-specific BERT model
    MODEL_NAME = config('MODEL_NAME', default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    # Backup models in order of preference
    BACKUP_MODELS = [
        'google/flan-t5-large',  # Better general language understanding
        'facebook/bart-large-cnn',  # Good at structured responses
        'microsoft/BioGPT-Large'  # Specialized for biomedical text
    ]
    MAX_TOKENS = config('MAX_TOKENS', default=500, cast=int)  # Increased for more detailed responses
    TEMPERATURE = config('TEMPERATURE', default=0.3, cast=float)  # Lower temperature for more focused responses
    APP_TITLE = config('APP_TITLE', default='AI Healthcare Assistant')
    DISCLAIMER = config('DISCLAIMER', default='⚠️ Always consult a healthcare provider for medical advice.')
    ENABLE_ANALYTICS = config('ENABLE_ANALYTICS', default=True, cast=bool)
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

# Medical knowledge base - key medical topics and their information
MEDICAL_KNOWLEDGE = {
    "diabetes": {
        "overview": "Diabetes is a chronic condition characterized by high levels of glucose in the blood due to inadequate insulin production or insulin resistance. There are several types of diabetes, with Type 1, Type 2, and gestational diabetes being the most common.",
        "symptoms": ["Increased thirst", "Frequent urination", "Unexplained weight loss", "Extreme hunger", "Blurred vision", "Fatigue", "Slow-healing sores", "Frequent infections", "Tingling in hands/feet"],
        "management": ["Regular blood glucose monitoring", "Medication adherence (insulin or oral medications)", "Balanced diet with carbohydrate monitoring", "Regular physical activity", "Regular medical check-ups", "Foot care", "Stress management"],
        "complications": ["Cardiovascular disease", "Neuropathy (nerve damage)", "Nephropathy (kidney damage)", "Retinopathy (eye damage)", "Foot damage that may lead to amputation", "Skin conditions", "Hearing impairment", "Alzheimer's disease"]
    },
    "hypertension": {
        "overview": "Hypertension, or high blood pressure, is a condition where the force of blood against artery walls is consistently too high. It's often called the 'silent killer' because it typically has no symptoms but significantly increases the risk of heart disease and stroke.",
        "symptoms": ["Often asymptomatic", "Headaches (in severe cases)", "Shortness of breath", "Nosebleeds", "Visual changes", "Dizziness", "Chest pain", "Blood in urine"],
        "management": ["Regular blood pressure monitoring", "Medication adherence", "Low sodium diet (less than 2,300mg daily)", "Regular physical activity (150 minutes per week)", "Stress management", "Limited alcohol consumption", "Weight management", "DASH diet approach"],
        "complications": ["Heart attack", "Stroke", "Heart failure", "Kidney damage or failure", "Vision loss", "Metabolic syndrome", "Vascular dementia", "Aneurysm"]
    },
    "asthma": {
        "overview": "Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to breathing difficulties. It affects the bronchial tubes that carry air to and from the lungs and can range from mild to severe.",
        "symptoms": ["Shortness of breath", "Chest tightness or pain", "Wheezing when exhaling", "Coughing, especially at night or early morning", "Difficulty sleeping due to breathing problems", "Fatigue", "Symptoms that worsen with respiratory infections"],
        "management": ["Identifying and avoiding triggers", "Using prescribed inhalers correctly (controller and rescue medications)", "Following asthma action plan", "Regular medical check-ups", "Breathing exercises", "Maintaining healthy weight", "Getting vaccinated against flu and pneumonia"],
        "complications": ["Severe asthma attacks requiring emergency care", "Respiratory failure", "Pneumonia and other respiratory infections", "Collapsed lung (pneumothorax)", "Airway remodeling (permanent narrowing)", "Side effects from medications"]
    },
    "heart_disease": {
        "overview": "Heart disease refers to several conditions affecting the heart, with coronary artery disease being the most common. It involves narrowed or blocked blood vessels that can lead to heart attack, chest pain, or stroke.",
        "symptoms": ["Chest pain, pressure or discomfort (angina)", "Shortness of breath", "Pain or numbness in extremities", "Pain in neck, jaw, throat, upper abdomen or back", "Fluttering in chest (palpitations)", "Racing heartbeat", "Slow heartbeat", "Lightheadedness", "Dizziness", "Fainting"],
        "management": ["Medication adherence", "Healthy diet low in saturated fats and sodium", "Regular physical activity", "Smoking cessation", "Limited alcohol consumption", "Stress management", "Weight management", "Regular health screenings"],
        "complications": ["Heart attack", "Heart failure", "Stroke", "Aneurysm", "Peripheral artery disease", "Sudden cardiac arrest", "Heart valve problems"]
    },
    "depression": {
        "overview": "Depression (major depressive disorder) is a common but serious mood disorder that causes persistent feelings of sadness and loss of interest. It affects how you feel, think, and handle daily activities and can lead to a variety of emotional and physical problems.",
        "symptoms": ["Persistent sad, anxious, or 'empty' mood", "Feelings of hopelessness or pessimism", "Irritability", "Feelings of guilt or worthlessness", "Loss of interest in hobbies and activities", "Decreased energy or fatigue", "Moving or talking slowly", "Difficulty concentrating or making decisions", "Insomnia or oversleeping", "Appetite or weight changes", "Thoughts of death or suicide"],
        "management": ["Professional therapy (cognitive behavioral therapy, interpersonal therapy)", "Medication (antidepressants)", "Regular physical activity", "Maintaining social connections", "Stress management techniques", "Sleep hygiene", "Avoiding alcohol and drugs", "Establishing routines"],
        "complications": ["Anxiety disorders", "Substance abuse", "Physical health problems", "Social isolation", "Self-harm or suicidal thoughts", "Work or school problems", "Relationship difficulties", "Reduced quality of life"]
    },
    "arthritis": {
        "overview": "Arthritis refers to inflammation of one or more joints, causing pain and stiffness that typically worsen with age. The two most common types are osteoarthritis (breakdown of cartilage) and rheumatoid arthritis (autoimmune disorder).",
        "symptoms": ["Joint pain", "Stiffness", "Swelling", "Redness", "Decreased range of motion", "Morning stiffness", "Fatigue", "Warmth in the joint", "Joint deformity (in severe cases)"],
        "management": ["Physical therapy", "Medication for pain and inflammation", "Weight management", "Regular, gentle exercise", "Hot and cold therapy", "Assistive devices", "Joint protection techniques", "Surgery in advanced cases"],
        "complications": ["Joint deformity", "Reduced mobility", "Difficulty performing daily tasks", "Work disability", "Metabolic disorders", "Cardiovascular disease (especially with rheumatoid arthritis)", "Mental health issues like depression"]
    }
}

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
    if not text or len(text.strip()) < 10:
        return "I apologize, but I need to provide a better answer. Please try asking your question again."

    # Remove any potential system message or prompt remnants
    text = re.sub(r'^(Assistant:|AI:|System:)', '', text, flags=re.IGNORECASE).strip()
    
    # Remove any template structures that might be echoed
    template_patterns = [
        r'1\.\s*Clear explanation of the medical topic.*?5\.\s*When to seek professional medical help',
        r'1\.\s*Comprehensive and well-structured.*?professional medical advice\.',
        r'Please provide a detailed response about this medical topic\.',
        r'Please provide a detailed, step-by-step response that includes:.*?When to seek professional medical help',
        r'Your responses should be:.*?professional medical advice\.',
        r'Remember to:.*?medical accuracy\.'
    ]
    
    for pattern in template_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    
    # Fix common formatting issues
    text = re.sub(r'\s*\n\s*\n\s*\n+', '\n\n', text)  # Remove excessive newlines
    text = re.sub(r'(?<=[.!?])\s*(?=[A-Z])', '\n\n', text)  # Add proper line breaks between sentences
    text = re.sub(r'(?m)^\s*[-•*]\s*', '• ', text)  # Standardize bullet points
    text = re.sub(r'(?m)^\s*(\d+)\.\s*', r'\1. ', text)  # Fix numbered lists
    
    # Ensure proper spacing
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\n\2', text)  # Add line breaks between sentences
    text = re.sub(r'([.!?])\s*(\d+\.)', r'\1\n\n\2', text)  # Add line breaks before numbered lists
    text = re.sub(r'([.!?])\s*(•)', r'\1\n\n\2', text)  # Add line breaks before bullet points
    
    # Fix common punctuation issues
    text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove spaces before punctuation
    text = re.sub(r'([.,!?])(?=[^\s])', r'\1 ', text)  # Add spaces after punctuation
    
    # Ensure proper list formatting
    lines = text.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        if line:
            # Check if line is a list item
            if re.match(r'^[•\-*]|^\d+\.', line):
                if not in_list:
                    formatted_lines.append('')  # Add space before list starts
                    in_list = True
            else:
                if in_list:
                    formatted_lines.append('')  # Add space after list ends
                    in_list = False
            formatted_lines.append(line)
    
    text = '\n'.join(formatted_lines).strip()
    
    # Add medical disclaimer if not present
    disclaimer_phrases = ['consult', 'healthcare provider', 'medical professional', 'doctor']
    if not any(phrase in text.lower() for phrase in disclaimer_phrases):
        text += "\n\nPlease note: This information is for educational purposes only. Always consult with a healthcare professional for personalized medical advice."
    
    # Final cleanup
    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive blank lines
    
    # If the text is too short after cleaning
    if len(text.strip()) < 50:
        return "I apologize, but I need to provide a more detailed answer. Please try asking your question again."
    
    return text

def enhance_medical_response(response, query):
    """Enhance the medical response with specific information from our knowledge base"""
    # Check if any key medical topics are mentioned in the query
    query_lower = query.lower()
    
    # If the response is already substantial (more than 500 characters), don't add too much more
    if len(response) > 500:
        add_details = False
    else:
        add_details = True
    
    # Look for medical terms in the query and response
    for topic, info in MEDICAL_KNOWLEDGE.items():
        if topic in query_lower or any(topic in response.lower() for topic in info["overview"].lower().split()[:3]):
            # Check what aspect of the topic is being asked
            if any(term in query_lower for term in ["symptom", "sign", "feel", "experience", "suffer"]):
                if add_details:
                    symptoms_text = "\n\nKey symptoms of " + topic + " include:\n" + "\n".join([f"• {symptom}" for symptom in info["symptoms"]])
                    if symptoms_text not in response:
                        response += symptoms_text
                elif not any(symptom.lower() in response.lower() for symptom in info["symptoms"][:2]):
                    # Even for long responses, add a brief mention if symptoms aren't covered
                    brief_symptoms = ", ".join(info["symptoms"][:3]) + ", and others"
                    if brief_symptoms not in response:
                        response += f"\n\nCommon symptoms include {brief_symptoms}."
            
            elif any(term in query_lower for term in ["manage", "treat", "control", "therapy", "cure", "medication", "drug", "medicine"]):
                if add_details:
                    management_text = "\n\nManagement approaches for " + topic + " include:\n" + "\n".join([f"• {approach}" for approach in info["management"]])
                    if management_text not in response:
                        response += management_text
                elif not any(approach.lower() in response.lower() for approach in info["management"][:2]):
                    brief_management = ", ".join(info["management"][:3]) + ", and other approaches"
                    if brief_management not in response:
                        response += f"\n\nKey management strategies include {brief_management}."
            
            elif any(term in query_lower for term in ["complication", "risk", "danger", "problem", "consequence"]):
                if add_details:
                    complications_text = "\n\nPossible complications of " + topic + " include:\n" + "\n".join([f"• {complication}" for complication in info["complications"]])
                    if complications_text not in response:
                        response += complications_text
                elif not any(complication.lower() in response.lower() for complication in info["complications"][:2]):
                    brief_complications = ", ".join(info["complications"][:3]) + ", and other complications"
                    if brief_complications not in response:
                        response += f"\n\nSerious complications can include {brief_complications}."
            
            elif any(term in query_lower for term in ["what is", "define", "overview", "about", "explain", "description"]):
                if info["overview"] not in response:
                    # Always include the overview if it's a definitional question
                    if not any(word in response.lower() for word in info["overview"].lower().split()[:10]):
                        response = info["overview"] + "\n\n" + response
            
            # If the query seems general, provide an overview and brief symptoms
            elif not any(specific in query_lower for specific in ["symptom", "manage", "treat", "complication", "risk", "what is"]):
                if not any(word in response.lower() for word in info["overview"].lower().split()[:10]):
                    response = info["overview"] + "\n\n" + response
                if add_details and not any(symptom.lower() in response.lower() for symptom in info["symptoms"][:2]):
                    brief_symptoms = ", ".join(info["symptoms"][:3]) + ", and others"
                    response += f"\n\nCommon symptoms include {brief_symptoms}."
    
    # If the response doesn't already have a note about seeing a healthcare professional, add one
    if not any(phrase in response.lower() for phrase in ['consult a healthcare', 'consult with a healthcare', 'consult your doctor', 'see a doctor', 'talk to your doctor']):
        response += "\n\nAlways consult with a healthcare professional for personalized medical advice and treatment options."
    
    return response

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
    
    # Medical topics detection
    topics = {
        'covid': {
            'keywords': ['covid', 'coronavirus', 'sars-cov-2', 'pandemic'],
            'subtopics': {
                'symptoms': ['symptom', 'sign', 'feel'],
                'prevention': ['prevent', 'protect', 'avoid'],
                'treatment': ['treat', 'cure', 'manage', 'medication', 'drug', 'therapy'],
                'testing': ['test', 'diagnose', 'detect'],
                'vaccine': ['vaccine', 'vaccination', 'shot', 'booster']
            }
        },
        'diabetes': {
            'keywords': ['diabetes', 'blood sugar', 'insulin', 'hyperglycemia', 'diabetic'],
            'subtopics': {
                'symptoms': ['symptom', 'sign', 'feel'],
                'management': ['manage', 'control', 'monitor', 'regulate'],
                'complications': ['complication', 'risk', 'danger', 'problem'],
                'prevention': ['prevent', 'avoid', 'reduce risk'],
                'types': ['type', 'kind', 'classification']
            }
        },
        'heart_health': {
            'keywords': ['heart', 'cardiac', 'cardiovascular', 'blood pressure', 'hypertension', 'cholesterol'],
            'subtopics': {
                'symptoms': ['symptom', 'sign', 'feel'],
                'prevention': ['prevent', 'avoid', 'reduce risk'],
                'risk_factors': ['risk', 'factor', 'cause', 'contribute'],
                'treatment': ['treat', 'manage', 'medication', 'drug', 'therapy'],
                'diagnosis': ['diagnose', 'test', 'detect', 'identify']
            }
        },
        'mental_health': {
            'keywords': ['mental health', 'anxiety', 'depression', 'stress', 'psychological', 'psychiatric'],
            'subtopics': {
                'symptoms': ['symptom', 'sign', 'feel'],
                'treatment': ['treat', 'therapy', 'counseling', 'medication'],
                'coping': ['cope', 'manage', 'deal with', 'self-care'],
                'resources': ['resource', 'help', 'support', 'service'],
                'types': ['type', 'kind', 'disorder', 'condition']
            }
        },
        'nutrition': {
            'keywords': ['nutrition', 'diet', 'food', 'eating', 'nutrient'],
            'subtopics': {
                'healthy_eating': ['healthy', 'balanced', 'nutritious'],
                'weight_management': ['weight', 'lose', 'gain', 'maintain'],
                'special_diets': ['vegan', 'vegetarian', 'keto', 'paleo', 'gluten-free', 'dairy-free'],
                'supplements': ['supplement', 'vitamin', 'mineral', 'protein'],
                'conditions': ['condition', 'disease', 'disorder', 'health']
            }
        },
        'exercise': {
            'keywords': ['exercise', 'workout', 'fitness', 'physical activity', 'sport'],
            'subtopics': {
                'cardio': ['cardio', 'aerobic', 'cardiovascular', 'running', 'jogging', 'swimming', 'cycling'],
                'strength': ['strength', 'resistance', 'weight', 'muscle', 'lifting'],
                'flexibility': ['flexibility', 'stretching', 'mobility', 'yoga', 'pilates'],
                'intensity': ['intensity', 'level', 'difficulty', 'hard', 'easy', 'moderate'],
                'benefits': ['benefit', 'advantage', 'health', 'improve']
            }
        },
        'sleep': {
            'keywords': ['sleep', 'insomnia', 'sleep apnea', 'snoring', 'rest', 'fatigue'],
            'subtopics': {
                'quality': ['quality', 'good', 'better', 'improve'],
                'disorders': ['disorder', 'problem', 'condition', 'issue'],
                'habits': ['habit', 'routine', 'hygiene', 'schedule'],
                'stages': ['stage', 'cycle', 'REM', 'deep', 'light'],
                'factors': ['factor', 'affect', 'influence', 'impact']
            }
        }
    }
    
    # Check if query matches any topics and subtopics
    for topic, data in topics.items():
        for keyword in data['keywords']:
            if keyword in query:
                main_topic = topic
                
                # Check for subtopics if main topic is identified
                if main_topic:
                    for sub, keywords in data['subtopics'].items():
                        for keyword in keywords:
                            if keyword in query:
                                subtopic = sub
                                break
                        if subtopic:
                            break
                break
        if main_topic:
            break
    
    return main_topic, subtopic, context

def get_model_response(prompt, model_name):
    """Get response from a specific model"""
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model_name}",
            headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "max_length": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "top_p": 0.95,
                    "do_sample": True,
                    "top_k": 50,
                    "repetition_penalty": 1.2,
                    "num_return_sequences": 1
                }
            },
            timeout=20
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            elif isinstance(result, dict):
                return result.get("generated_text", "")
    except Exception as e:
        st.warning(f"Error with model {model_name}: {str(e)}")
    return None

def get_chatbot_response(user_input, uploaded_file=None):
    """Get a response using HuggingFace's inference API or fallback to a rule-based system"""
    try:
        # Prepare the prompt with system message and context
        system_message = """You are a medical AI assistant trained to provide accurate healthcare information. 
Focus on providing:
1. Clear, evidence-based medical information
2. Specific symptoms, causes, and treatments when relevant
3. Prevention and management strategies
4. References to medical guidelines when applicable
5. Clear indicators for when to seek professional medical help

Important: Maintain medical accuracy while being accessible to general audience."""
        
        # Build conversation history
        conversation_history = []
        for message in st.session_state.messages[-3:]:
            prefix = "User: " if message["role"] == "user" else "Assistant: "
            conversation_history.append(f"{prefix}{message['content']}")
        
        conversation_str = "\n".join(conversation_history)
        
        # Prepare content
        content = user_input
        if uploaded_file is not None:
            file_content = extract_text_from_file(uploaded_file)
            content += f"\n\nContext from uploaded document: {file_content}"
            
        # Create the full prompt
        full_prompt = f"""{system_message}

Previous Conversation:
{conversation_str}

User Question: {content}

Provide a detailed medical response:"""

        if HUGGINGFACE_API_KEY:
            # Try primary model first
            st.info("Generating medical response...")
            response = get_model_response(full_prompt, MODEL_NAME)
            
            if response:
                formatted_response = format_ai_response(response)
                if len(formatted_response) > 50:
                    return enhance_medical_response(formatted_response, content)
            
            # Try backup models in sequence
            for backup_model in BACKUP_MODELS:
                st.info(f"Trying alternative medical model...")
                response = get_model_response(full_prompt, backup_model)
                
                if response:
                    formatted_response = format_ai_response(response)
                    if len(formatted_response) > 50:
                        return enhance_medical_response(formatted_response, content)
            
            st.warning("Models unavailable. Using built-in knowledge system.")
        else:
            st.info("No API key provided. Using built-in healthcare knowledge system.")
        
        # Fallback to rule-based system
        fallback_response = generate_fallback_response(content)
        return enhance_medical_response(fallback_response, content)
        
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        return "I apologize, but I'm having trouble accessing my knowledge base. Please try again in a moment."

def generate_fallback_response(query):
    """Generate a response using the built-in healthcare knowledge system"""
    main_topic, subtopic, context = analyze_query(query)
    
    # Base responses for different topics and subtopics
    responses = {
        'covid': {
            'symptoms': [
                "Common COVID-19 symptoms include fever, cough, fatigue, and loss of taste or smell. Most symptoms appear 2-14 days after exposure. {severity} {history}",
                "COVID-19 typically presents with respiratory symptoms like shortness of breath, sore throat, and nasal congestion, along with fever and fatigue. {severity} {history}",
                "Initial COVID-19 signs often include fever, dry cough, and tiredness, which may progress to more severe symptoms in some cases. {severity} {history}"
            ],
            'prevention': [
                "Key COVID-19 prevention measures include staying up-to-date with vaccines, wearing masks in crowded indoor settings, maintaining good hand hygiene, and ensuring adequate ventilation. {temporal}",
                "To prevent COVID-19: wash hands frequently with soap for at least 20 seconds, wear well-fitting masks in public settings, maintain social distance, and get vaccinated and boosted as recommended. {temporal}",
                "Protect yourself from COVID-19 by getting vaccinated, avoiding crowded poorly ventilated spaces, washing hands regularly, and monitoring your health for symptoms. {temporal}"
            ],
            'treatment': [
                "COVID-19 treatment depends on severity. Mild cases can be managed at home with rest, hydration, and over-the-counter fever reducers. Several antiviral medications are available for those at high risk. {severity} {demographic}",
                "For most people with mild COVID-19, supportive care focused on symptom relief is sufficient. High-risk individuals may benefit from antiviral treatments if started early. {severity} {demographic}",
                "Treatment options for COVID-19 now include oral antivirals like Paxlovid for high-risk patients, while severe cases may require hospitalization, oxygen therapy, or other interventions. {severity} {demographic}"
            ]
        },
        'diabetes': {
            'symptoms': [
                "Common diabetes symptoms include increased thirst, frequent urination, unexplained weight loss, extreme hunger, blurred vision, fatigue, and slow-healing sores. {severity} {history}",
                "Type 1 diabetes typically develops rapidly with symptoms including excessive thirst, frequent urination, extreme hunger, and unintended weight loss despite increased appetite. {severity} {history}",
                "Type 2 diabetes signs often develop gradually and may include increased thirst and urination, fatigue, blurred vision, numbness in extremities, and recurring infections. {severity} {history}"
            ],
            'management': [
                "Diabetes management involves regular blood sugar monitoring, medication adherence, consistent carbohydrate intake, regular physical activity, and stress management. Working with a healthcare team to create a personalized plan is essential. {temporal}",
                "Effective diabetes control requires monitoring blood glucose levels regularly, taking medications as prescribed, maintaining consistent meal timing and content, engaging in regular exercise, and attending scheduled medical check-ups. {temporal}",
                "Managing diabetes successfully involves a combination of blood glucose monitoring, medication management, healthy eating patterns, regular physical activity, and routine medical care with specialists including endocrinologists and certified diabetes educators. {temporal}"
            ],
            'complications': [
                "Diabetes complications can affect multiple body systems, including cardiovascular disease, nerve damage (neuropathy), kidney damage (nephropathy), eye damage (retinopathy), and foot problems that may lead to serious infections. {severity} {history}",
                "Long-term complications of diabetes include increased risk of heart attack, stroke, vision problems, kidney failure, and nerve damage leading to pain or numbness, particularly in the extremities. {severity} {history}",
                "Poorly controlled diabetes can lead to serious complications including cardiovascular disease, kidney disease requiring dialysis, vision loss, nerve damage, and foot problems that may lead to amputation. Tight glucose control helps reduce these risks. {severity} {history}"
            ]
        },
        'mental_health': {
            'symptoms': [
                "Common mental health symptoms include persistent changes in mood, sleep patterns, energy levels, concentration, appetite, and interest in previously enjoyed activities. {severity} {history}",
                "Mental health concerns often manifest as anxiety, persistent sadness, irritability, social withdrawal, difficulty concentrating, and changes in sleep and eating patterns. {severity} {history}",
                "Watch for changes in daily functioning, mood, energy levels, sleep patterns, appetite, concentration, and social engagement as potential signs of mental health issues. {severity} {history}"
            ],
            'treatment': [
                "Mental health treatment typically involves a combination of psychotherapy (like cognitive-behavioral therapy), sometimes medication, lifestyle changes, social support, and self-care practices tailored to the individual's specific condition and needs. {demographic}",
                "Evidence-based mental health treatments include various forms of therapy (such as CBT, DBT, or IPT), psychiatric medications when appropriate, support groups, mindfulness practices, and lifestyle modifications addressing sleep, nutrition, and exercise. {demographic}",
                "Treatment plans for mental health conditions are personalized and may combine different therapeutic approaches including individual therapy, group therapy, medication management, stress-reduction techniques, and addressing lifestyle factors like sleep hygiene and physical activity. {demographic}"
            ],
            'coping': [
                "Healthy coping strategies for mental health include regular physical activity, maintaining social connections, practicing mindfulness or meditation, getting adequate sleep, limiting alcohol and caffeine, and seeking professional help when needed. {temporal}",
                "To cope with mental health challenges, consider developing a routine that includes physical activity, relaxation techniques, adequate sleep, healthy eating, limited social media exposure, and regular social interaction with supportive people. {temporal}",
                "Effective coping mechanisms include practicing deep breathing exercises, progressive muscle relaxation, journaling, engaging in enjoyable activities, maintaining social connections, setting realistic goals, and establishing healthy boundaries. {temporal}"
            ]
        },
        'nutrition': {
            'healthy_eating': [
                "A balanced diet should include a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats while limiting added sugars, sodium, and highly processed foods. Aim for colorful, diverse foods at each meal. {temporal}",
                "Focus on eating whole, unprocessed foods including a rainbow of fruits and vegetables, whole grains like brown rice and quinoa, lean proteins such as fish and legumes, and healthy fats from sources like avocados, nuts, and olive oil. {temporal}",
                "Healthy eating involves consuming appropriate portions of nutrient-dense foods including plenty of fruits, vegetables, whole grains, lean proteins, and healthy fats, while staying hydrated and limiting foods high in added sugars, sodium, and unhealthy fats. {temporal}"
            ],
            'weight_management': [
                "Sustainable weight management involves balanced nutrition, regular physical activity, adequate sleep, stress management, and behavioral strategies rather than restrictive short-term diets. Small, consistent changes are more effective long-term. {demographic}",
                "Focus on creating healthy, sustainable habits such as eating mindfully, incorporating more plant foods, staying physically active, managing stress, getting enough sleep, and addressing emotional eating rather than pursuing rapid weight loss. {demographic}",
                "Consider tracking your food intake and physical activity to identify patterns, set specific and realistic goals, incorporate both cardiovascular exercise and strength training, manage portion sizes, and celebrate non-scale victories. {demographic}"
            ],
            'special_diets': [
                "When following a {context} diet, ensure you're meeting all nutritional requirements by consulting with a healthcare provider or registered dietitian who can help identify potential nutrient gaps and recommend appropriate supplements if needed. {temporal}",
                "Special diets should be planned carefully to avoid nutrient deficiencies. Focus on maximizing food variety within the diet's parameters and consider working with a nutrition professional to ensure all nutrient needs are met. {temporal}",
                "Consider consulting with a registered dietitian for personalized dietary advice when following special diets to ensure nutritional adequacy, sustainability, and appropriateness for your specific health conditions and goals. {temporal}"
            ]
        },
        'heart_health': {
            'prevention': [
                "Maintain heart health by consuming a diet rich in fruits, vegetables, whole grains, and healthy fats while limiting sodium, saturated fats, and added sugars. Regular physical activity, stress management, and avoiding smoking are also essential. {temporal}",
                "Regular cardiovascular exercise (at least 150 minutes of moderate activity weekly), strength training, maintaining healthy blood pressure and cholesterol levels, and avoiding tobacco products all support long-term heart health. {temporal}",
                "Prevent heart disease by managing key risk factors including high blood pressure, high cholesterol, diabetes, obesity, physical inactivity, poor diet, excessive alcohol consumption, and tobacco use. Regular check-ups help detect problems early. {temporal}"
            ],
            'symptoms': [
                "Watch for heart problem warning signs like chest pain/discomfort (which may feel like pressure, squeezing, or fullness), shortness of breath, pain/discomfort in the arms, back, neck, jaw or stomach, and cold sweats, nausea, or lightheadedness. {severity} {history}",
                "Heart attack symptoms often include chest discomfort (pressure, squeezing, fullness), discomfort in other upper body areas, shortness of breath, and sometimes cold sweat, nausea, or lightheadedness. Women may experience less obvious symptoms. {severity} {history}",
                "Some heart conditions may cause fatigue, dizziness, irregular heartbeat (palpitations), swelling in the legs/ankles/feet, persistent cough, or shortness of breath during daily activities or when lying down. {severity} {history}"
            ],
            'risk_factors': [
                "Key heart disease risk factors include high blood pressure, high cholesterol, diabetes, obesity, physical inactivity, unhealthy diet, smoking, excessive alcohol consumption, stress, and family history. Many factors can be modified through lifestyle changes. {demographic}",
                "Family history, age, and sex affect heart disease risk (men generally develop it earlier than women), but modifiable factors like smoking, diet, physical activity, and managing conditions like hypertension have a significant impact. {demographic}",
                "Lifestyle choices significantly impact heart health. Managing stress, getting adequate sleep, maintaining a healthy weight, exercising regularly, eating a heart-healthy diet, and avoiding tobacco can substantially reduce heart disease risk. {demographic}"
            ]
        },
        'exercise': {
            'cardio': [
                "Aim for at least 150 minutes of moderate cardio activity (like brisk walking) or 75 minutes of vigorous activity (like running) per week, spread across multiple days for optimal cardiovascular benefits. {demographic}",
                "Choose activities you enjoy such as walking, swimming, cycling, dancing, or group fitness classes to maintain motivation and consistency with your cardiovascular exercise routine. {demographic}",
                "Start gradually with 10-15 minute sessions if you're new to cardio exercise and increase duration and intensity as your fitness improves, monitoring how your body responds and adjusting accordingly. {demographic}"
            ],
            'strength': [
                "Include strength training 2-3 times per week, targeting all major muscle groups and allowing at least 48 hours of recovery between sessions for a specific muscle group. Progressive overload is key to continued improvements. {temporal}",
                "Focus on major muscle groups and proper form during exercises, starting with lighter weights and mastering technique before increasing resistance. Compound movements like squats, deadlifts, and push-ups provide efficient full-body benefits. {temporal}",
                "Allow adequate rest between strength training sessions as muscle growth and repair occurs during recovery. Ensure proper nutrition with adequate protein intake to support muscle development and recovery. {temporal}"
            ],
            'flexibility': [
                "Regular stretching after warming up muscles can improve flexibility, range of motion, and posture while reducing injury risk and muscle tension. Hold static stretches for 15-30 seconds without bouncing. {severity}",
                "Consider incorporating yoga, Pilates, or tai chi into your routine to improve flexibility while also enhancing strength, balance, and mind-body awareness. These practices offer comprehensive physical and mental benefits. {severity}",
                "Stretch major muscle groups daily, especially after exercise when muscles are warm. Focus on areas of tightness particular to your body and activities, and remember that consistency is more important than intensity. {severity}"
            ]
        },
        'sleep': {
            'quality': [
                "Good sleep quality involves both sufficient duration (7-9 hours for most adults) and consistency in sleep timing, with minimal nighttime awakenings and feeling refreshed upon waking. {temporal}",
                "Create a relaxing bedtime routine to improve sleep quality, such as disconnecting from screens 1-2 hours before bed, engaging in calming activities like reading or gentle stretching, and keeping your bedroom cool, dark, and quiet. {temporal}",
                "Aim for 7-9 hours of uninterrupted sleep each night, maintaining consistent sleep and wake times even on weekends to support your body's natural circadian rhythm and hormone regulation. {temporal}"
            ],
            'disorders': [
                "Sleep disorders can significantly impact overall health, affecting cardiovascular health, immune function, cognitive performance, mental health, and metabolic processes. Early identification and treatment are important. {severity} {history}",
                "Common sleep disorders include insomnia (difficulty falling or staying asleep), sleep apnea (breathing interruptions during sleep), restless leg syndrome (uncomfortable sensations causing an urge to move the legs), and narcolepsy (excessive daytime sleepiness). {severity} {history}",
                "Consider a sleep study if you experience persistent sleep issues like excessive daytime sleepiness, loud snoring, gasping or choking during sleep, significant difficulty falling or staying asleep, or unusual behaviors during sleep. {severity} {history}"
            ],
            'habits': [
                "Maintain consistent sleep and wake times, even on weekends, to help regulate your body's internal clock. This consistency reinforces your sleep-wake cycle and can help you fall asleep more quickly and sleep more soundly. {temporal}",
                "Avoid screens (phones, tablets, computers, TV) at least 30-60 minutes before bedtime as the blue light can suppress melatonin production. Also limit caffeine, alcohol, and large meals close to bedtime as they can disrupt sleep quality. {temporal}",
                "Create a sleep-friendly environment that's dark (use blackout curtains if needed), quiet (consider earplugs or white noise if necessary), and cool (around 65°F/18°C). Your mattress, pillows, and bedding should also provide comfort and support. {temporal}"
            ]
        }
    }
    
    # Select appropriate response base
    if main_topic in responses and subtopic in responses[main_topic]:
        response_templates = responses[main_topic][subtopic]
        base_response = random.choice(response_templates)
    else:
        # Fallback to general response if topic/subtopic not found
        general_responses = [
            "I understand you're asking about {topic}. While I don't have specific information on this exact query, I recommend consulting with a healthcare professional for personalized advice tailored to your situation.",
            "Your question about {topic} is important. For the most accurate and personalized information, I'd recommend discussing this with a healthcare provider who can consider your complete medical history and specific circumstances.",
            "Regarding your {topic} question, the best approach would be to consult with a qualified healthcare professional who can provide guidance specific to your individual health needs and circumstances."
        ]
        base_response = random.choice(general_responses)
    
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
        general_advice="I recommend consulting with a healthcare professional for personalized advice.",
        context=context.get('demographic', '')
    )
    
    # Add a disclaimer
    response += "\n\nPlease note: This information is for educational purposes only. Always consult with a healthcare professional for personalized medical advice."
    
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
    """Extract text content from uploaded files (TXT, PDF, DOCX)"""
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        text_content = ""
        
        if file_type == 'txt':
            # Text files
            text_content = uploaded_file.getvalue().decode('utf-8')
        
        elif file_type == 'pdf':
            # PDF files
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() + "\n"
            except Exception as e:
                st.error(f"Error reading PDF: {str(e)}")
                return "Could not extract text from the PDF file."
        
        elif file_type == 'docx':
            # Word documents
            try:
                doc = docx.Document(uploaded_file)
                for para in doc.paragraphs:
                    text_content += para.text + "\n"
            except Exception as e:
                st.error(f"Error reading DOCX: {str(e)}")
                return "Could not extract text from the Word document."
        
        else:
            return "Unsupported file format. Please upload a TXT, PDF, or DOCX file."
        
        # Extract important medical keywords
        medical_keywords = extract_medical_keywords(text_content)
        
        # Format the returned text
        if len(text_content) > 1000:
            summary = text_content[:1000] + "...\n\n"
            if medical_keywords:
                summary += "Key medical concepts detected: " + ", ".join(medical_keywords)
            return summary
        else:
            if medical_keywords:
                text_content += "\n\nKey medical concepts detected: " + ", ".join(medical_keywords)
            return text_content
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return "Error processing the uploaded file."

def extract_medical_keywords(text):
    """Extract important medical keywords and concepts from text"""
    # List of common medical terms to look for
    medical_terms = {
        "conditions": ["diabetes", "hypertension", "asthma", "cancer", "stroke", "heart attack", 
                      "alzheimer", "parkinson", "arthritis", "depression", "anxiety", "copd", 
                      "pneumonia", "bronchitis", "influenza", "covid", "obesity", "insomnia"],
        
        "symptoms": ["pain", "fatigue", "fever", "cough", "headache", "nausea", "vomiting", 
                    "dizziness", "shortness of breath", "chest pain", "abdominal pain", 
                    "rash", "swelling", "inflammation", "bleeding", "numbness", "tingling"],
        
        "treatments": ["surgery", "medication", "therapy", "antibiotics", "vaccine", "chemotherapy", 
                      "radiation", "transplant", "dialysis", "physical therapy", "immunotherapy"],
        
        "vitals": ["blood pressure", "heart rate", "pulse", "temperature", "respiratory rate", 
                  "oxygen saturation", "glucose", "cholesterol"]
    }
    
    found_terms = []
    
    # Clean and lowercase the text
    text_lower = text.lower()
    
    # Look for medical terms in the text
    for category, terms in medical_terms.items():
        for term in terms:
            if term in text_lower:
                found_terms.append(term)
    
    # Look for measurements and values (e.g., "120/80 mmHg", "98.6°F")
    bp_matches = re.findall(r'\b\d{2,3}/\d{2,3}\b', text)  # Blood pressure format
    temp_matches = re.findall(r'\b\d{2}\.\d°[FC]\b|\b\d{2,3}°[FC]\b', text)  # Temperature format
    
    # Add found measurements
    for match in bp_matches:
        found_terms.append(f"blood pressure {match}")
    
    for match in temp_matches:
        found_terms.append(f"temperature {match}")
    
    # Return unique terms
    return list(set(found_terms))

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
                    if st.button("👍", key=f"like_{i}", help="Helpful response"):
                        st.session_state.feedback_given[i] = "positive"
                        log_interaction(
                            st.session_state.messages[i-1]["content"] if i > 0 else "", 
                            message["content"], 
                            "positive"
                        )
                        st.markdown(
                            """
                            <style>
                            [data-testid="baseButton-secondary"]:has(div:contains("👍")) {
                                color: #28a745 !important;
                            }
                            </style>
                            """,
                            unsafe_allow_html=True
                        )
                        st.rerun()
                with col2:
                    if st.button("👎", key=f"dislike_{i}", help="Unhelpful response"):
                        st.session_state.feedback_given[i] = "negative"
                        log_interaction(
                            st.session_state.messages[i-1]["content"] if i > 0 else "", 
                            message["content"], 
                            "negative"
                        )
                        st.markdown(
                            """
                            <style>
                            [data-testid="baseButton-secondary"]:has(div:contains("👎")) {
                                color: #dc3545 !important;
                            }
                            </style>
                            """,
                            unsafe_allow_html=True
                        )
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
