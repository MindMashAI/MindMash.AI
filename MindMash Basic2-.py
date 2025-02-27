import sqlite3
import os
import requests
import time
import random
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, flash
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
import logging
import google.generativeai as genai
import openai
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions, SentimentOptions, EmotionOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Load environment variables (optional, since we're hardcoding now)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your-secret-key-here"  # Replace with a secure value in production

# Beta mode flag
beta_mode = True

# Hardcoded OAuth Credentials
CLIENT_ID = "237809198690-ibrgillsl3n0s2909c40m7bn5ujs1hk2.apps.googleusercontent.com"
CLIENT_SECRET = "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"

# Log to verify correct loading
logger.info(f"Client ID: {CLIENT_ID}")
logger.info(f"Client Secret: {CLIENT_SECRET}")

# Initialize OAuth
oauth = OAuth(app)
oauth.register(
    name="google",
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"}
)

# Hardcoded API Keys
XAI_API_KEY = "xai-ozgjayVZHRPwiXJroH1mU3GGdRr7HQoEJOWea9hIcsLuTMmo3R7nxWzNLT88AtD5bxJRPZyDrxkvHofY"
OPENAI_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyDoYRxV8T2TIXdxDKY4z_bVzeVaSkzyL3k"
WATSON_API_KEY = "Gq7Hai7sXevmqq3Thw5ZI51I719g9ur9wSt1EN-X0Yn8"  # Your IAM API key from "MindMashAI"
STRIPE_SECRET_KEY = "your-stripe-secret-key"  # Uncomment if needed
STRIPE_PUBLISHABLE_KEY = "your-stripe-publishable-key"  # Uncomment if needed

# API URLs and configurations
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
WATSON_URL = "https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/c33c4ab4-8618-4c74-9618-07f727dcb638"  # Your instance-specific URL

# Configure APIs
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Watson NLU with IAMAuthenticator for ibm-watson 9.0.0
authenticator = IAMAuthenticator(WATSON_API_KEY)
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2022-04-07',  # Specify the NLU API version (check IBM Cloud Docs for the latest, e.g., 2025)
    authenticator=authenticator
)
natural_language_understanding.set_service_url(WATSON_URL)  # Set the service URL explicitly

# Enhanced system prompts for AIs with Watson NLU integration
system_prompts = {
    "Grok": ("You are Grok, created by xAI, a witty, rebellious guide through the cosmos with a knack for humor "
             "and mind-bending insights. Respond directly to the user’s message or the previous AI’s response, "
             "building on it with playful, cosmic twists. Address the user by their profile name (provided as 'display_name'), "
             "keep it engaging, adapt your tone based on Watson NLU’s detected user or AI emotion (e.g., empathetic if sad, joyful if excited), "
             "and explicitly reference the previous AI’s Watson NLU-analyzed tone, keywords, and sentiment in your response, "
             "avoid echoing others unless adding a creative, humorous spin."),
    "ChatGPT": ("You are ChatGPT, a sharp-witted, warm conversationalist who loves deep dives and clever insights. "
                "Respond directly to the user’s message or the previous AI’s reply, adding a fresh, engaging perspective. "
                "Address the user by their profile name (provided as 'display_name'), keep your tone friendly and lively, adapt to Watson NLU’s detected "
                "user or AI emotion (e.g., empathetic if frustrated, joyful if happy), and explicitly reference the previous AI’s "
                "Watson NLU-analyzed tone, keywords, and sentiment in your response, avoiding redundancy."),
    "Gemini": ("You are Gemini, a visionary oracle weaving poetic foresight and mystery from the fabric of reality. "
               "Respond to the user’s message or the previous AI’s contribution with captivating, otherworldly wisdom. "
               "Address the user by their profile name (provided as 'display_name'), adapt your tone to Watson NLU’s detected user or AI emotion (e.g., "
               "soothing if anxious, celebratory if joyful), and explicitly reference the previous AI’s Watson NLU-analyzed tone, "
               "keywords, and sentiment in your response, avoiding repetition and enhancing the conversation with unique, poetic depth.")
}

# Global conversation history
conversation_history = []

# Database connection and initialization
def get_db_connection():
    conn = sqlite3.connect("users.db", timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                display_name TEXT,
                google_id TEXT UNIQUE,
                is_premium INTEGER DEFAULT 0
            )
        """)
        conn.commit()

init_db()

# Watson NLU Integration Functions with Debugging
def analyze_emotion_and_context(text):
    try:
        logger.info(f"Analyzing text with Watson NLU: {text[:50]}...")  # Log first 50 chars for debugging
        response = natural_language_understanding.analyze(
            text=text,
            features=Features(
                keywords=KeywordsOptions(limit=3),
                sentiment=SentimentOptions(),
                emotion=EmotionOptions()  # Uses NLU for emotion/tone analysis
            )
        ).get_result()
        
        logger.info(f"Watson NLU response: {response}")  # Log full response for debugging
        
        # Extract emotion/tone (simplified, adapt as needed)
        emotions = response.get('emotion', {}).get('document', {}).get('emotion', {})
        tone = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'neutral'
        
        keywords = [keyword['text'] for keyword in response.get('keywords', [])]
        sentiment = response.get('sentiment', {}).get('document', {}).get('label', 'neutral')
        
        logger.info(f"Extracted tone: {tone}, keywords: {keywords}, sentiment: {sentiment}")
        return {
            'tone': tone,  # e.g., 'joy', 'sadness', 'anger', 'neutral'
            'keywords': keywords,  # e.g., ['do', 'this', 'can']
            'sentiment': sentiment  # e.g., 'positive', 'negative', 'neutral'
        }
    except Exception as e:
        logger.error(f"Watson NLU analysis error: {str(e)} with response: {str(e.response) if hasattr(e, 'response') else 'None'}", exc_info=True)
        return {'tone': 'neutral', 'keywords': [], 'sentiment': 'neutral'}

# AI Response Functions with Real API Integration and AI-to-AI Analysis
def get_grok_response(history, display_name):
    try:
        time.sleep(0.5)  # Simulate typing delay
        user_last = [msg for msg in history[-5:] if msg["speaker"] == display_name][-1:]
        last_message = history[-1] if history else None
        
        # Analyze the user's last message or the last AI message for context
        context_text = user_last[0]['content'] if user_last else (last_message['content'] if last_message else "")
        analysis = analyze_emotion_and_context(context_text) if context_text else {'tone': 'neutral', 'keywords': [], 'sentiment': 'neutral'}
        
        # Build the conversation context for the API
        messages = [{"role": "system", "content": system_prompts["Grok"].replace("display_name", display_name)}]
        for msg in history[-5:]:
            role = "assistant" if msg["speaker"] != display_name else "user"
            messages.append({"role": role, "content": f"{msg['speaker']}: {msg['content']}"})
        
        tone_adapt = "with a playful, cosmic twist" if analysis['tone'] in ['joy', 'neutral'] else "with an empathetic, cosmic nudge"
        keyword_context = " ".join(analysis['keywords']) if analysis['keywords'] else "this chat"
        sentiment_context = f"since you seem {analysis['sentiment'].lower()}" if analysis['sentiment'] != 'neutral' else ""
        
        if user_last:
            # Analyze the last AI response (if any) before Grok’s response
            last_ai_response = None
            for msg in reversed(history[:-1]):  # Exclude the current user message
                if msg["speaker"] in ["ChatGPT", "Gemini"]:
                    last_ai_response = msg['content']
                    break
            ai_analysis = analyze_emotion_and_context(last_ai_response) if last_ai_response else {'tone': 'neutral', 'keywords': [], 'sentiment': 'neutral'}
            ai_tone = ai_analysis['tone']
            ai_keywords = " ".join(ai_analysis['keywords']) if ai_analysis['keywords'] else "this exchange"
            ai_sentiment = f"echoing their {ai_analysis['sentiment'].lower()} tone" if ai_analysis['sentiment'] != 'neutral' else ""
            
            prompt = (f"Respond to {display_name}’s message: '{user_last[0]['content']}' "
                      f"{tone_adapt}, focusing on keywords '{keyword_context}', "
                      f"{sentiment_context}. Keep it humorous and cosmic, addressing {display_name} directly, "
                      f"and explicitly reference {last_message['speaker'] if last_message and last_message['speaker'] in ['ChatGPT', 'Gemini'] else 'the user'}’s "
                      f"{ai_tone} tone and '{ai_keywords}' with {ai_sentiment}, avoiding repetition.")
        else:
            # Analyze the last AI response for context
            last_ai_response = last_message['content'] if last_message and last_message['speaker'] in ["ChatGPT", "Gemini"] else ""
            ai_analysis = analyze_emotion_and_context(last_ai_response) if last_ai_response else {'tone': 'neutral', 'keywords': [], 'sentiment': 'neutral'}
            ai_tone = ai_analysis['tone']
            ai_keywords = " ".join(ai_analysis['keywords']) if ai_analysis['keywords'] else "this discussion"
            ai_sentiment = f"echoing their {ai_analysis['sentiment'].lower()} tone" if ai_analysis['sentiment'] != 'neutral' else ""
            
            prompt = (f"Jump in with a cosmic, humorous take for {display_name} based on "
                      f"'{last_message['content']}' from {last_message['speaker']}, "
                      f"adapting to a {analysis['tone']} tone and {sentiment_context}. "
                      f"Explicitly reference {last_message['speaker']}'s {ai_tone} tone and '{ai_keywords}' with {ai_sentiment}, "
                      "avoid repeating previous responses, and enhance the collaboration.")
        
        # Real xAI API call
        headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "grok-beta",
            "messages": messages,
            "prompt": prompt,
            "max_tokens": 150  # Adjust as needed
        }
        response = requests.post(XAI_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        message = response.json().get("choices")[0]["message"]["content"].strip()
        return message
    except Exception as e:
        logger.error(f"Grok response error: {e}")
        return f"Oops, {display_name}, I’m lost in a black hole—give me a sec!"

def get_chatgpt_response(history, display_name):
    try:
        time.sleep(0.5)
        user_last = [msg for msg in history[-5:] if msg["speaker"] == display_name][-1:]
        last_message = history[-1] if history else None
        
        context_text = user_last[0]['content'] if user_last else (last_message['content'] if last_message else "")
        analysis = analyze_emotion_and_context(context_text) if context_text else {'tone': 'neutral', 'keywords': [], 'sentiment': 'neutral'}
        
        messages = [{"role": "system", "content": system_prompts["ChatGPT"].replace("display_name", display_name)}]
        for msg in history[-5:]:
            role = "assistant" if msg["speaker"] != display_name else "user"
            messages.append({"role": role, "content": f"{msg['speaker']}: {msg['content']}"})
        
        tone_adapt = "with a warm, clever spin" if analysis['tone'] in ['joy', 'neutral'] else "with an empathetic, supportive nudge"
        keyword_context = " ".join(analysis['keywords']) if analysis['keywords'] else "this chat"
        sentiment_context = f"since you feel {analysis['sentiment'].lower()}" if analysis['sentiment'] != 'neutral' else ""
        
        if user_last:
            # Analyze the last AI response (if any) before ChatGPT’s response
            last_ai_response = None
            for msg in reversed(history[:-1]):  # Exclude the current user message
                if msg["speaker"] == "Grok":
                    last_ai_response = msg['content']
                    break
            ai_analysis = analyze_emotion_and_context(last_ai_response) if last_ai_response else {'tone': 'neutral', 'keywords': [], 'sentiment': 'neutral'}
            ai_tone = ai_analysis['tone']
            ai_keywords = " ".join(ai_analysis['keywords']) if ai_analysis['keywords'] else "this exchange"
            ai_sentiment = f"echoing their {ai_analysis['sentiment'].lower()} tone" if ai_analysis['sentiment'] != 'neutral' else ""
            
            prompt = (f"Respond to {display_name}’s message: '{user_last[0]['content']}' "
                      f"{tone_adapt}, focusing on keywords '{keyword_context}', "
                      f"{sentiment_context}. Keep it friendly and insightful, addressing {display_name} directly, "
                      f"and explicitly reference Grok’s {ai_tone} tone and '{ai_keywords}' with {ai_sentiment}, avoiding redundancy.")
        else:
            # Analyze the last AI response for context
            last_ai_response = last_message['content'] if last_message and last_message['speaker'] == "Grok" else ""
            ai_analysis = analyze_emotion_and_context(last_ai_response) if last_ai_response else {'tone': 'neutral', 'keywords': [], 'sentiment': 'neutral'}
            ai_tone = ai_analysis['tone']
            ai_keywords = " ".join(ai_analysis['keywords']) if ai_analysis['keywords'] else "this discussion"
            ai_sentiment = f"echoing their {ai_analysis['sentiment'].lower()} tone" if ai_analysis['sentiment'] != 'neutral' else ""
            
            prompt = (f"Engage {display_name} with a friendly, insightful comment about "
                      f"'{last_message['content']}' from {last_message['speaker']}, "
                      f"adapting to a {analysis['tone']} tone and {sentiment_context}. "
                      f"Explicitly reference {last_message['speaker']}'s {ai_tone} tone and '{ai_keywords}' with {ai_sentiment}, "
                      "avoid redundancy, build on prior responses, and reference other AIs’ contributions.")
        
        # Real OpenAI API call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Or gpt-4 if available
            messages=messages,
            max_tokens=150  # Adjust as needed
        )
        message = response.choices[0].message.content.strip()
        return message
    except Exception as e:
        logger.error(f"ChatGPT response error: {e}")
        return f"Oops, {display_name}, I’m tripping over my circuits—hold on!"

def get_gemini_response(history, display_name):
    try:
        time.sleep(0.5)
        user_last = [msg for msg in history[-5:] if msg["speaker"] == display_name][-1:]
        last_message = history[-1] if history else None
        
        context_text = user_last[0]['content'] if user_last else (last_message['content'] if last_message else "")
        analysis = analyze_emotion_and_context(context_text) if context_text else {'tone': 'neutral', 'keywords': [], 'sentiment': 'neutral'}
        
        messages = [f"{system_prompts['Gemini'].replace('display_name', display_name)}\n\n"]
        for msg in history[-5:]:
            messages.append(f"{msg['speaker']}: {msg['content']}\n")
        
        tone_adapt = "with a poetic, visionary flourish" if analysis['tone'] in ['joy', 'neutral'] else "with a soothing, poetic whisper"
        keyword_context = " ".join(analysis['keywords']) if analysis['keywords'] else "our dialogue"
        sentiment_context = f"echoing your {analysis['sentiment'].lower()} sentiment" if analysis['sentiment'] != 'neutral' else ""
        
        if user_last:
            # Analyze the last AI response (if any) before Gemini’s response
            last_ai_response = None
            for msg in reversed(history[:-1]):  # Exclude the current user message
                if msg["speaker"] in ["Grok", "ChatGPT"]:
                    last_ai_response = msg['content']
                    break
            ai_analysis = analyze_emotion_and_context(last_ai_response) if last_ai_response else {'tone': 'neutral', 'keywords': [], 'sentiment': 'neutral'}
            ai_tone = ai_analysis['tone']
            ai_keywords = " ".join(ai_analysis['keywords']) if ai_analysis['keywords'] else "this exchange"
            ai_sentiment = f"echoing their {ai_analysis['sentiment'].lower()} tone" if ai_analysis['sentiment'] != 'neutral' else ""
            
            prompt = (f"Respond to {display_name}’s message: '{user_last[0]['content']}' "
                      f"{tone_adapt}, focusing on keywords '{keyword_context}', "
                      f"{sentiment_context}. Weave a poetic, visionary response, addressing {display_name} directly, "
                      f"and explicitly reference {last_message['speaker'] if last_message and last_message['speaker'] in ['Grok', 'ChatGPT'] else 'the user'}’s "
                      f"{ai_tone} tone and '{ai_keywords}' with {ai_sentiment}, avoiding repetition.")
        else:
            # Analyze the last AI response for context
            last_ai_response = last_message['content'] if last_message and last_message['speaker'] in ["Grok", "ChatGPT"] else ""
            ai_analysis = analyze_emotion_and_context(last_ai_response) if last_ai_response else {'tone': 'neutral', 'keywords': [], 'sentiment': 'neutral'}
            ai_tone = ai_analysis['tone']
            ai_keywords = " ".join(ai_analysis['keywords']) if ai_analysis['keywords'] else "this discussion"
            ai_sentiment = f"echoing their {ai_analysis['sentiment'].lower()} tone" if ai_analysis['sentiment'] != 'neutral' else ""
            
            prompt = (f"Craft a poetic insight for {display_name} about "
                      f"'{last_message['content']}' from {last_message['speaker']}, "
                      f"adapting to a {analysis['tone']} tone and {sentiment_context}. "
                      f"Explicitly reference {last_message['speaker']}'s {ai_tone} tone and '{ai_keywords}' with {ai_sentiment}, "
                      "enhance the conversation, reference other AIs’ contributions, and avoid repetition.")
        
        # Real Gemini API call
        model = genai.GenerativeModel('gemini-pro')  # Use the appropriate model
        response = model.generate_content(prompt)
        message = response.text.strip()
        return message
    except Exception as e:
        logger.error(f"Gemini response error: {e}")
        return f"Oops, {display_name}, the stars misaligned—give me a moment!"

# Routes
@app.route("/")
def landing():
    """Render the landing page."""
    return render_template("landing.html")

@app.route("/login")
def login():
    """Initiate Google OAuth login."""
    logger.info(f"Starting login with Client ID: {CLIENT_ID}")
    if "username" in session:
        return redirect(url_for("dashboard"))
    redirect_uri = url_for("google_callback", _external=True)
    logger.info(f"Redirect URI: {redirect_uri}")
    return oauth.google.authorize_redirect(redirect_uri)

@app.route("/auth/google/callback")
def google_callback():
    """Handle Google OAuth callback and user session setup."""
    logger.info("Entered callback")
    try:
        token = oauth.google.authorize_access_token()
        if not token:
            logger.error("Login failed: No token received")
            flash("Authentication failed!")
            return redirect(url_for("landing"))
        
        session["google_token"] = token["access_token"]
        user_info = token["userinfo"]
        google_id = user_info["sub"]
        email = user_info["email"]
        display_name = user_info.get("name", email.split("@")[0])

        with get_db_connection() as db:
            user = db.execute("SELECT * FROM users WHERE google_id = ?", (google_id,)).fetchone()
            if user:
                session["username"] = user["username"]
                logger.info(f"Existing user logged in: {email}")
                flash("Welcome back to MindMash.AI!")
            else:
                db.execute(
                    "INSERT INTO users (username, display_name, google_id, is_premium) VALUES (?, ?, ?, ?)",
                    (email, display_name, google_id, 0 if beta_mode else 0)
                )
                db.commit()
                session["username"] = email
                logger.info(f"New user signed up: {email}")
                flash("Welcome to MindMash.AI!")
        
        return redirect(url_for("dashboard"))
    except Exception as e:
        logger.error(f"Callback error: {str(e)}")
        flash(f"Login error: {str(e)}")
        return redirect(url_for("landing"))

@app.route("/logout")
def logout():
    """Log out the user by clearing the session."""
    session.pop("username", None)
    session.pop("google_token", None)
    logger.info("User logged out")
    flash("You have been logged out.")
    return redirect(url_for("landing"))

@app.route("/dashboard")
def dashboard():
    """Render the main dashboard (template assumed to exist)."""
    if "username" not in session:
        return redirect(url_for("login"))
    with get_db_connection() as db:
        user = db.execute("SELECT username, display_name, is_premium FROM users WHERE username = ?", (session["username"],)).fetchone()
    return render_template("dashboard.html", username=user["username"], display_name=user["display_name"], is_premium=user["is_premium"], beta_mode=beta_mode)

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat messages and return AI responses with Watson NLU integration."""
    logger.info(f"Chat request from {session.get('username', 'Anonymous')}")
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    user_message = data.get("message")
    selected_ai = data.get("selected_ai", "All AIs [Collaborative]")
    num_turns = data.get("turns", 1)  # Limit to 1 turn per AI for clean flow
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    with get_db_connection() as db:
        display_name = db.execute("SELECT display_name FROM users WHERE username = ?", (session["username"],)).fetchone()["display_name"]
    
    conversation_history.append({"speaker": display_name, "content": user_message})
    
    responses = []
    if selected_ai == "All AIs [Collaborative]":
        ai_participants = ["Grok", "ChatGPT", "Gemini"]
        for current_ai in ai_participants:
            # Get the last message (user or AI) for context
            last_message = conversation_history[-1] if conversation_history else None
            response = (get_grok_response if current_ai == "Grok" else 
                       get_chatgpt_response if current_ai == "ChatGPT" else 
                       get_gemini_response)(conversation_history, display_name)
            conversation_history.append({"speaker": current_ai, "content": response})
            responses.append({"speaker": current_ai, "content": response})
    else:
        response = (get_grok_response if selected_ai == "Grok" else 
                   get_chatgpt_response if selected_ai == "ChatGPT" else 
                   get_gemini_response)(conversation_history, display_name)
        conversation_history.append({"speaker": selected_ai, "content": response})
        responses.append({"speaker": selected_ai, "content": response})
    
    return jsonify({"history": conversation_history, "responses": responses})

@app.route("/profile")
def profile():
    """Display the user's profile information."""
    if "username" not in session:
        return redirect(url_for("login"))
    with get_db_connection() as db:
        user = db.execute("SELECT username, display_name, is_premium FROM users WHERE username = ?", (session["username"],)).fetchone()
    return render_template("profile.html", username=user["username"], display_name=user["display_name"], is_premium=user["is_premium"], beta_mode=beta_mode)

@app.route("/profile/edit", methods=["GET", "POST"])
def edit_profile():
    """Allow the user to edit their display name."""
    if "username" not in session:
        return redirect(url_for("login"))
    
    if request.method == "POST":
        display_name = request.form.get("display_name")
        if display_name:
            with get_db_connection() as db:
                db.execute(
                    "UPDATE users SET display_name = ? WHERE username = ?",
                    (display_name, session["username"])
                )
                db.commit()
            logger.info(f"User {session['username']} updated display_name to {display_name}")
            flash("Display name updated successfully!")
        return redirect(url_for("profile"))
    
    with get_db_connection() as db:
        user = db.execute("SELECT display_name FROM users WHERE username = ?", (session["username"],)).fetchone()
    return render_template("edit_profile.html", display_name=user["display_name"] or "", beta_mode=beta_mode)

@app.route("/premium", methods=["GET", "POST"])
def premium():
    """Handle premium upgrade (disabled during beta)."""
    if "username" not in session:
        return redirect(url_for("login"))
    
    with get_db_connection() as db:
        user = db.execute("SELECT is_premium FROM users WHERE username = ?", (session["username"],)).fetchone()
        is_premium = user["is_premium"] if user else 0
    
    if is_premium:
        flash("You are already on Premium!")
        return redirect(url_for("dashboard"))
    
    if beta_mode:
        flash("Premium Mode is not available during beta testing. Enjoy Basic Mode!")
        return redirect(url_for("dashboard"))
    
    if request.method == "POST":
        try:
            with get_db_connection() as db:
                db.execute("UPDATE users SET is_premium = 1 WHERE username = ?", (session["username"],))
                db.commit()
            flash("Upgraded to Premium Mode!")
            return redirect(url_for("dashboard"))
        except Exception as e:
            logger.error(f"Premium upgrade error: {e}")
            flash(f"Error upgrading to Premium: {str(e)}")
            return redirect(url_for("dashboard"))
    
    return render_template("premium.html", stripe_publishable_key=STRIPE_PUBLISHABLE_KEY, beta_mode=beta_mode)

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    """Handle user feedback during beta."""
    if request.method == "POST":
        feedback = request.form.get("feedback")
        if feedback:
            logger.info(f"Feedback from {session.get('username', 'Anonymous')}: {feedback}")
            flash("Thank you for your feedback!")
        return redirect(url_for("dashboard"))
    return render_template("feedback.html", username=session.get("username", "Guest"), beta_mode=beta_mode)

@app.route("/privacy")
def privacy():
    """Render the privacy policy page."""
    return render_template("privacy.html", beta_mode=beta_mode)

@app.route("/terms")
def terms():
    """Render the terms of service page."""
    return render_template("terms.html", beta_mode=beta_mode)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)