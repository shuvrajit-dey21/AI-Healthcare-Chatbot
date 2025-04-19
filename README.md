# AI Healthcare Assistant

A modern, interactive healthcare chatbot powered by AI that provides healthcare information and assistance in a conversational manner.

![AI Healthcare Assistant](https://via.placeholder.com/800x400?text=AI+Healthcare+Assistant)

## Features

- 🤖 AI-powered healthcare information using HuggingFace's models with fallback capabilities
- 💬 Interactive chat interface with bubble styling and session memory
- 🎙️ Voice input capability for hands-free interaction
- 📁 File upload to provide context from health documents with medical keyword extraction
- 🌓 Light/dark theme toggle for comfortable viewing
- 📊 Optional usage analytics to track interactions
- 🚨 Emergency keyword detection with appropriate warnings
- 👍👎 User feedback collection to improve responses
- 📤 Export chat history for record keeping
- 🏥 Enhanced medical knowledge base for offline responses
- 🔄 Multi-model fallback system for reliable responses

## Quick Start

### Prerequisites

- Python 3.8 or higher
- HuggingFace API key (optional, the app works without it using the fallback system)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI-Healthcare-Chatbot.git
   cd AI-Healthcare-Chatbot
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root directory with your configuration:
   ```env
   HUGGINGFACE_API_KEY=your_huggingface_api_key_here
   MODEL_NAME=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
   BACKUP_MODEL=google/flan-t5-base
   MAX_TOKENS=350
   TEMPERATURE=0.4
   APP_TITLE=AI Healthcare Assistant
   DISCLAIMER=⚠️ Always consult a healthcare provider for medical advice.
   ENABLE_ANALYTICS=true
   ANALYTICS_FILE=usage_analytics.json
   ```
   
   Note: The HuggingFace API key is optional. If not provided, the system will use a built-in fallback mechanism to generate responses.

### Running the Application

Run the application with Streamlit:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` in your web browser.

## Usage Guide

### Basic Chat
- Type your healthcare-related question in the input field
- Click the "Submit" button to get a response
- The chat history is preserved during your session

### Voice Input
1. Click the "🎤 Speak your question" button in the sidebar
2. Speak clearly when prompted
3. Your speech will be converted to text and added to the input field

### Document Upload
1. Use the file uploader in the sidebar to upload health-related documents (PDF, DOCX, TXT)
2. The content of the documents will be used to provide context for the AI's responses
3. The system will automatically extract key medical concepts from your document to improve responses

### Export Chat
1. Click the "📤 Export Chat" button in the sidebar
2. Download the chat transcript as a text file

### Toggle Theme
- Click the "Toggle Light/Dark Theme" button to switch between light and dark modes

## Response Generation

The application uses multiple methods for generating responses:

1. **Primary Model (BiomedNLP-PubMedBERT)** - If a HuggingFace API key is provided, the app will first try to use this specialized medical language model for generating accurate, medical-focused responses.

2. **Backup Model (Flan-T5)** - If the primary model is unavailable or times out, the system automatically falls back to this general-purpose model, which is optimized for conversation.

3. **Enhanced Medical Knowledge Base** - All responses are augmented with specific medical information from our built-in knowledge base on topics like diabetes, hypertension, and asthma.

4. **Rule-based Fallback System** - If both models are unavailable, the app will use a comprehensive rule-based system with detailed medical information to generate responses for common healthcare topics.

## Important Notes

- **This is not a replacement for professional medical advice**
- The AI may not have access to the latest medical research
- Always consult with a qualified healthcare provider for medical concerns
- In case of emergency, call emergency services immediately

## Analytics

If analytics are enabled in the `.env` file (`ENABLE_ANALYTICS=true`), the application will log:
- Timestamp of interactions
- User inputs (for quality improvement)
- Response lengths
- User feedback

This data is stored locally in the file specified by `ANALYTICS_FILE` and can be visualized in the sidebar when analytics are enabled.

## Development

### Project Structure
```
AI-Healthcare-Chatbot/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (not in git)
├── .env.example        # Example environment variables
├── README.md           # This file
└── .gitignore          # Git ignore file
```

### Extending the Application

- To add new features, modify the `app.py` file
- To change the UI, adjust the CSS in the `get_chat_bubble_css()` function
- To modify the AI behavior, update the system message in the `get_chatbot_response()` function
- To expand the medical knowledge base, add more entries to the `MEDICAL_KNOWLEDGE` dictionary
- To enhance the fallback system, add more rules to the `generate_fallback_response()` function

## Recent Improvements

- Added a specialized medical language model (BiomedNLP-PubMedBERT) for more accurate medical responses
- Implemented a multi-model fallback system for better reliability
- Enhanced the built-in medical knowledge base for comprehensive offline responses
- Added medical keyword extraction from uploaded documents
- Improved response quality with more detailed, evidence-based medical information
- Optimized API calls with faster timeouts and better error handling
- Enhanced analytics capabilities for tracking system performance

## Deployment Options

### Deploy to Streamlit Cloud

The easiest way to deploy this application is using Streamlit Cloud:

1. Push your code to a GitHub repository
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Configure your app settings and deploy

### Deploy with Docker

You can also deploy using Docker:

1. Create a Dockerfile in your project directory:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "app.py"]
   ```

2. Build and run the Docker container:
   ```bash
   docker build -t healthcare-chatbot .
   docker run -p 8501:8501 healthcare-chatbot
   ```

### Deploy on Heroku

To deploy on Heroku:

1. Create a `Procfile` in your project directory:
   ```
   web: streamlit run app.py
   ```

2. Push to Heroku:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

3. Set environment variables in Heroku dashboard or using CLI:
   ```bash
   heroku config:set HUGGINGFACE_API_KEY=your_key_here
   ```

### Important Deployment Notes

- Make sure to set up environment variables properly in your deployment environment
- For PyAudio to work in production, additional system dependencies may be required
- Some cloud providers may have limitations with the voice input feature

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Powered by HuggingFace's language models
- Built with Streamlit
- Speech recognition using Google's Speech Recognition API

## Disclaimer

This application is for informational purposes only and is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. 