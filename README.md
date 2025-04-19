# AI Healthcare Assistant

A modern, interactive healthcare chatbot powered by AI that provides healthcare information and assistance in a conversational manner.

![AI Healthcare Assistant](https://via.placeholder.com/800x400?text=AI+Healthcare+Assistant)

## Features

- 🤖 AI-powered healthcare information using HuggingFace's models with fallback capabilities
- 💬 Interactive chat interface with bubble styling and session memory
- 🎙️ Voice input capability for hands-free interaction
- 📁 File upload to provide context from health documents
- 🌓 Light/dark theme toggle for comfortable viewing
- 📊 Optional usage analytics to track interactions
- 🚨 Emergency keyword detection with appropriate warnings
- 👍👎 User feedback collection to improve responses
- 📤 Export chat history for record keeping

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
   MODEL_NAME=google/flan-t5-base
   MAX_TOKENS=150
   TEMPERATURE=0.7
   APP_TITLE=AI Healthcare Assistant
   DISCLAIMER=⚠️ Always consult a healthcare provider for medical advice.
   ENABLE_ANALYTICS=false
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
1. Use the file uploader in the sidebar to upload health-related documents
2. The content of the documents will be used to provide context for the AI's responses

### Export Chat
1. Click the "📤 Export Chat" button in the sidebar
2. Download the chat transcript as a text file

### Toggle Theme
- Click the "Toggle Light/Dark Theme" button to switch between light and dark modes

## Response Generation

The application uses two methods for generating responses:

1. **HuggingFace Inference API** - If a HuggingFace API key is provided, the app will use the specified model to generate responses. This provides the most advanced and up-to-date responses.

2. **Fallback Response System** - If the HuggingFace API is unavailable or no API key is provided, the app will use a built-in rule-based system to generate responses for common healthcare topics. This ensures the app remains functional even without internet access or API keys.

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

This data is stored locally in the file specified by `ANALYTICS_FILE`.

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
- To enhance the fallback system, add more rules to the `generate_fallback_response()` function

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