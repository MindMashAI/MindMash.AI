import sqlite3
import os
import requests
import time
import random
import json
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, flash
from flask_socketio import SocketIO, emit
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
import logging
import google.generativeai as genai
import openai
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions, SentimentOptions, EmotionOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask app with SocketIO
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default-secret-key")  # Use environment variable
socketio = SocketIO(app)  # Start with default configuration (no CORS or async_mode initially)

# Beta mode flag
beta_mode = True

# OAuth Credentials (loaded from .env)
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    logger.error("Google OAuth credentials are missing. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env.")

logger.info(f"Client ID: {CLIENT_ID}")

# Initialize OAuth
oauth = OAuth(app)
oauth.register(
    name="google",
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"}
)

# API Keys (loaded from .env)
XAI_API_KEY = os.getenv("XAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WATSON_API_KEY = os.getenv("WATSON_API_KEY")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY")

# API URLs and configurations
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
WATSON_URL = os.getenv("WATSON_URL")

# Configure APIs
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize Watson NLU
if WATSON_API_KEY:
    authenticator = IAMAuthenticator(WATSON_API_KEY)
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version="2022-04-07",
        authenticator=authenticator
    )
    natural_language_understanding.set_service_url(WATSON_URL)
else:
    logger.warning("Watson API Key is missing. Natural Language Understanding service will not be initialized.")

# System prompts for AI collaboration with role-specific adjustments
system_prompts = {
    "Grok": ("You’re Grok, built by xAI—a sharp, friendly voice with a cosmic perspective. "
             "Respond directly to the user’s message in a clear, casual way, using their profile name ('display_name'). "
             "Adapt your tone based on the Watson NLU analysis and short-term memory (last 10 messages). "
             "Avoid repeating recent content; bring a unique twist."),
    "ChatGPT": ("You’re ChatGPT—a warm, insightful buddy focused on synthesis. Respond directly to the user’s "
                "message in a friendly way, using their profile name ('display_name'). Tune your tone to the "
                "Watson NLU analysis and short-term memory (last 10 messages). Offer a fresh angle."),
    "Gemini": ("You’re Gemini—a calm, forward-thinking presence. Respond directly to the user’s message "
               "in a grounded way, using their profile name ('display_name'). Adjust your tone to the Watson NLU "
               "analysis and short-term memory (last 10 messages). Provide a unique perspective.")
}

# Global conversation and emotion history
conversation_history = []
emotion_history = []

# Database connection and initialization
def get_db_connection():
    conn = sqlite3.connect("users.db", timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        # Check if table exists and add new columns if needed
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                display_name TEXT,
                google_id TEXT UNIQUE,
                is_premium INTEGER DEFAULT 0
            )
        """)
        # Add pinned_messages and tone_preference if they don't exist
        try:
            conn.execute("ALTER TABLE users ADD COLUMN pinned_messages TEXT DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass  # Column already exists, no action needed
        try:
            conn.execute("ALTER TABLE users ADD COLUMN tone_preference TEXT DEFAULT 'casual'")
        except sqlite3.OperationalError:
            pass  # Column already exists, no action needed
        conn.commit()

init_db()

# Watson NLU Analysis with intent detection
def analyze_emotion_and_context(text):
    try:
        response = natural_language_understanding.analyze(
            text=text,
            features=Features(
                keywords=KeywordsOptions(limit=3),
                sentiment=SentimentOptions(),
                emotion=EmotionOptions()
            )
        ).get_result()
        
        emotions = response.get('emotion', {}).get('document', {}).get('emotion', {})
        tone = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'neutral'
        keywords = [keyword['text'] for keyword in response.get('keywords', [])]
        sentiment = response.get('sentiment', {}).get('document', {}).get('label', 'neutral')
        sentiment_score = response.get('sentiment', {}).get('document', {}).get('score', 0.0)
        
        # Intent detection
        intent = 'general'
        text_lower = text.lower()
        if any(word in text_lower for word in ['technical', 'code', 'data', 'algorithm']):
            intent = 'technical'
        elif any(word in text_lower for word in ['philosophical', 'ethics', 'meaning', 'purpose']):
            intent = 'philosophical'
        elif any(word in text_lower for word in ['future', 'vision', 'innovation', 'trend']):
            intent = 'visionary'
        
        return {
            'emotion': tone,
            'keywords': keywords,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'emotion_scores': emotions,
            'intent': intent
        }
    except Exception as e:
        logger.error(f"Watson NLU analysis error: {str(e)}")
        return {'emotion': 'neutral', 'keywords': [], 'sentiment': 'neutral', 'sentiment_score': 0.0, 'emotion_scores': {}, 'intent': 'general'}

# AI Response Functions with Dynamic Roles and Tone Adjustment
def get_grok_response(context_package):
    try:
        time.sleep(0.5)
        display_name = context_package["display_name"]
        user_message = context_package["user_message"]
        watson_analysis = context_package["watson_analysis"]
        short_term_memory = context_package["short_term_memory"]
        pinned_messages = context_package.get("pinned_messages", [])
        tone_preference = context_package.get("tone_preference", "casual")
        intent = watson_analysis["intent"]
        trend = watson_analysis.get("trend", "stable")
        
        messages = [{"role": "system", "content": system_prompts["Grok"].replace("display_name", display_name)}]
        for msg in short_term_memory[-10:]:
            messages.append({"role": "user" if msg["speaker"] == display_name else "assistant", 
                             "content": f"{msg['speaker']}: {msg['content']}"})
        
        prompt = f"Respond to {display_name}’s message: '{user_message}'. "
        if intent == 'philosophical':
            prompt += "As the lead for this philosophical discussion, provide your cosmic perspective. "
        if pinned_messages:
            prompt += "Remember these pinned messages: " + " ".join(pinned_messages) + ". "
        prompt += f"Use a {tone_preference} tone. The user's sentiment trend is {trend}. Adjust accordingly."
        
        headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "grok-beta",
            "messages": messages,
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.8
        }
        response = requests.post(XAI_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        content = response.json().get("choices")[0]["message"]["content"].strip()
        return content
    except Exception as e:
        logger.error(f"Grok response error: {e}")
        return f"Hey {display_name}, I hit a snag—hang tight!"

def get_chatgpt_response(context_package):
    try:
        time.sleep(0.5)
        display_name = context_package["display_name"]
        user_message = context_package["user_message"]
        watson_analysis = context_package["watson_analysis"]
        short_term_memory = context_package["short_term_memory"]
        pinned_messages = context_package.get("pinned_messages", [])
        tone_preference = context_package.get("tone_preference", "casual")
        intent = watson_analysis["intent"]
        trend = watson_analysis.get("trend", "stable")
        
        messages = [{"role": "system", "content": system_prompts["ChatGPT"].replace("display_name", display_name)}]
        for msg in short_term_memory[-10:]:
            messages.append({"role": "user" if msg["speaker"] == display_name else "assistant", 
                             "content": f"{msg['speaker']}: {msg['content']}"})
        
        prompt = f"Respond to {display_name}’s message: '{user_message}'. "
        if intent == 'technical':
            prompt += "As the lead for this technical discussion, provide a synthesized view. "
        if pinned_messages:
            prompt += "Remember these pinned messages: " + " ".join(pinned_messages) + ". "
        prompt += f"Use a {tone_preference} tone. The user's sentiment trend is {trend}. Adjust accordingly."
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages + [{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"ChatGPT response error: {e}")
        return f"Hey {display_name}, I hit a snag—hang tight!"

def get_gemini_response(context_package):
    try:
        time.sleep(0.5)
        display_name = context_package["display_name"]
        user_message = context_package["user_message"]
        watson_analysis = context_package["watson_analysis"]
        short_term_memory = context_package["short_term_memory"]
        pinned_messages = context_package.get("pinned_messages", [])
        tone_preference = context_package.get("tone_preference", "casual")
        intent = watson_analysis["intent"]
        trend = watson_analysis.get("trend", "stable")
        
        messages = [system_prompts["Gemini"].replace("display_name", display_name) + "\n"]
        for msg in short_term_memory[-10:]:
            messages.append(f"{msg['speaker']}: {msg['content']}\n")
        
        prompt = f"Respond to {display_name}’s message: '{user_message}'. "
        if intent == 'visionary':
            prompt += "As the lead for this visionary discussion, provide your forward-thinking perspective. "
        if pinned_messages:
            prompt += "Remember these pinned messages: " + " ".join(pinned_messages) + ". "
        prompt += f"Use a {tone_preference} tone. The user's sentiment trend is {trend}. Adjust accordingly."
        
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini response error: {e}")
        return f"Hey {display_name}, I hit a snag—hang tight!"

# SocketIO Event Handlers with Enhanced Interactivity
@socketio.on('chat_message')
def handle_chat_message(data):
    if "username" not in session:
        emit('chat_response', {'type': 'system', 'message': 'Unauthorized'})
        return
    
    user_message = data.get("message")
    selected_ai = data.get("ai", "All AIs [Collaborative]")
    turns = min(int(data.get("turns", 1)), 10)
    
    if not user_message:
        emit('chat_response', {'type': 'system', 'message': 'No message provided'})
        return
    
    with get_db_connection() as db:
        user = db.execute("SELECT display_name, pinned_messages, tone_preference FROM users WHERE username = ?", 
                         (session["username"],)).fetchone()
        display_name = user["display_name"]
        pinned_messages = json.loads(user["pinned_messages"]) if user["pinned_messages"] else []
        tone_preference = user["tone_preference"]
    
    conversation_history.append({"speaker": display_name, "content": user_message})
    emit('chat_response', {'type': 'ai', 'speaker': display_name, 'message': user_message, 'emotion': 'neutral'})
    
    watson_analysis = analyze_emotion_and_context(user_message)
    
    # Emotional continuity tracking
    if len(emotion_history) >= 2:
        avg_sentiment = sum(e['sentiment_score'] for e in emotion_history[-5:]) / 5
        current_score = watson_analysis['sentiment_score']
        if current_score > avg_sentiment + 0.1:
            watson_analysis['trend'] = 'increasing positivity'
        elif current_score < avg_sentiment - 0.1:
            watson_analysis['trend'] = 'increasing negativity'
        else:
            watson_analysis['trend'] = 'stable'
    else:
        watson_analysis['trend'] = 'stable'
    emotion_history.append(watson_analysis)
    
    context_package = {
        "display_name": display_name,
        "user_message": user_message,
        "watson_analysis": watson_analysis,
        "short_term_memory": conversation_history[-10:],
        "pinned_messages": pinned_messages,
        "tone_preference": tone_preference
    }
    
    ai_order = {
        'technical': ['ChatGPT', 'Grok', 'Gemini'],
        'philosophical': ['Grok', 'ChatGPT', 'Gemini'],
        'visionary': ['Gemini', 'Grok', 'ChatGPT'],
        'general': ['Grok', 'ChatGPT', 'Gemini']
    }.get(watson_analysis['intent'], ['Grok', 'ChatGPT', 'Gemini'])
    
    if selected_ai == "All AIs [Collaborative]":
        for turn in range(turns):
            for ai in ai_order:
                emit('chat_response', {'type': 'typing', 'speaker': ai})
                response_func = {'Grok': get_grok_response, 'ChatGPT': get_chatgpt_response, 'Gemini': get_gemini_response}[ai]
                response = response_func(context_package)
                analysis = analyze_emotion_and_context(response)
                conversation_history.append({"speaker": ai, "content": response})
                emit('chat_response', {'type': 'ai', 'speaker': ai, 'message': response, 'emotion': analysis['emotion']})
                context_package["short_term_memory"] = conversation_history[-10:]
            
            # Self-awareness summary every 10 messages
            if len(conversation_history) >= 10 and len(conversation_history) % 10 == 0:
                summary_prompt = "Summarize the recent conversation, highlighting key points and suggesting next steps."
                model = genai.GenerativeModel('gemini-1.5-pro')
                summary = model.generate_content(summary_prompt).text.strip()
                conversation_history.append({"speaker": "Gemini", "content": summary})
                emit('chat_response', {'type': 'ai', 'speaker': "Gemini", 'message': summary, 'emotion': 'neutral'})
        
        emit('chat_response', {'type': 'system', 'message': 'Which AI helped most? Click their name on the map.'})
    else:
        response_func = {'Grok': get_grok_response, 'ChatGPT': get_chatgpt_response, 'Gemini': get_gemini_response}[selected_ai]
        for _ in range(turns):
            emit('chat_response', {'type': 'typing', 'speaker': selected_ai})
            response = response_func(context_package)
            analysis = analyze_emotion_and_context(response)
            conversation_history.append({"speaker": selected_ai, "content": response})
            emit('chat_response', {'type': 'ai', 'speaker': selected_ai, 'message': response, 'emotion': analysis['emotion']})
            context_package["short_term_memory"] = conversation_history[-10:]
        emit('chat_response', {'type': 'system', 'message': 'Which AI helped most? Click their name on the map.'})
    
    # Update collaboration map
    nodes = [{'id': speaker, 'color': {'Grok': '#00ffff', 'ChatGPT': '#cc00ff', 'Gemini': '#00ff00', display_name: '#ffffff'}.get(speaker, '#ffffff')}
             for speaker in set(msg['speaker'] for msg in conversation_history[-10:])]
    links = [{"source": display_name, "target": ai} for ai in ai_order if any(msg['speaker'] == ai for msg in conversation_history[-10:])]
    emit('map_interactivity', {'nodes': nodes, 'links': links}, broadcast=True)

@socketio.on('pin_message')
def handle_pin_message(data):
    message_content = data.get('message')
    if message_content and "username" in session:
        with get_db_connection() as db:
            user = db.execute("SELECT pinned_messages FROM users WHERE username = ?", (session["username"],)).fetchone()
            pinned_messages = json.loads(user["pinned_messages"]) if user["pinned_messages"] else []
            if message_content not in pinned_messages:
                pinned_messages.append(message_content)
                db.execute("UPDATE users SET pinned_messages = ? WHERE username = ?", 
                          (json.dumps(pinned_messages), session["username"]))
                db.commit()

@socketio.on('set_tone_preference')
def handle_tone_preference(data):
    tone = data.get('tone')
    if tone in ['casual', 'formal', 'technical'] and "username" in session:
        with get_db_connection() as db:
            db.execute("UPDATE users SET tone_preference = ? WHERE username = ?", (tone, session["username"]))
            db.commit()

@socketio.on('submit_feedback')
def handle_feedback(data):
    chosen_ai = data.get('chosen_ai')
    if chosen_ai and "username" in session:
        logger.info(f"User {session['username']} chose {chosen_ai} as most helpful")
        # Future: Adjust AI weights here (e.g., in a future update)

@socketio.on('node_click')
def handle_node_click(data):
    speaker = data.get('speaker')
    if speaker and speaker in ['Grok', 'ChatGPT', 'Gemini', session.get('display_name', 'User')]:
        last_message = next((msg['content'] for msg in reversed(conversation_history) if msg['speaker'] == speaker), 
                           "No recent messages.")
        emit('chat_response', {'type': 'system', 'message': f"Last from {speaker}: {last_message}"})

@socketio.on('node_drag')
def handle_node_drag(data):
    speaker = data.get('speaker')
    x = data.get('x')
    y = data.get('y')
    if speaker and x is not None and y is not None:
        emit('map_update', {'speaker': speaker, 'x': x, 'y': y}, broadcast=True)

# Routes
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/login")
def login():
    if "username" in session:
        return redirect(url_for("dashboard"))
    redirect_uri = url_for("google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route("/auth/google/callback")
def google_callback():
    try:
        token = oauth.google.authorize_access_token()
        user_info = token["userinfo"]
        google_id = user_info["sub"]
        email = user_info["email"]
        display_name = user_info.get("name", email.split("@")[0])

        with get_db_connection() as db:
            user = db.execute("SELECT * FROM users WHERE google_id = ?", (google_id,)).fetchone()
            if user:
                session["username"] = user["username"]
            else:
                db.execute(
                    "INSERT INTO users (username, display_name, google_id, is_premium) VALUES (?, ?, ?, ?)",
                    (email, display_name, google_id, 0 if beta_mode else 0)
                )
                db.commit()
                session["username"] = email
        return redirect(url_for("dashboard"))
    except Exception as e:
        logger.error(f"Callback error: {str(e)}")
        flash(f"Login error: {str(e)}")
        return redirect(url_for("landing"))

@app.route("/logout")
def logout():
    session.pop("username", None)
    flash("You have been logged out.")
    return redirect(url_for("landing"))

@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    with get_db_connection() as db:
        user = db.execute("SELECT username, display_name, is_premium FROM users WHERE username = ?", (session["username"],)).fetchone()
    return render_template("dashboard.html", username=user["username"], display_name=user["display_name"], is_premium=user["is_premium"], beta_mode=beta_mode)

@app.route("/profile")
def profile():
    if "username" not in session:
        return redirect(url_for("login"))
    with get_db_connection() as db:
        user = db.execute("SELECT username, display_name, is_premium FROM users WHERE username = ?", (session["username"],)).fetchone()
    return render_template("profile.html", username=user["username"], display_name=user["display_name"], is_premium=user["is_premium"], beta_mode=beta_mode)

@app.route("/profile/edit", methods=["GET", "POST"])
def edit_profile():
    if "username" not in session:
        return redirect(url_for("login"))
    
    if request.method == "POST":
        display_name = request.form.get("display_name")
        if display_name:
            with get_db_connection() as db:
                db.execute("UPDATE users SET display_name = ? WHERE username = ?", (display_name, session["username"]))
                db.commit()
            flash("Display name updated successfully!")
        return redirect(url_for("profile"))
    
    with get_db_connection() as db:
        user = db.execute("SELECT display_name FROM users WHERE username = ?", (session["username"],)).fetchone()
    return render_template("edit_profile.html", display_name=user["display_name"] or "", beta_mode=beta_mode)

@app.route("/premium", methods=["GET", "POST"])
def premium():
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
    if request.method == "POST":
        feedback = request.form.get("feedback")
        if feedback:
            logger.info(f"Feedback from {session.get('username', 'Anonymous')}: {feedback}")
            flash("Thank you for your feedback!")
        return redirect(url_for("dashboard"))
    return render_template("feedback.html", username=session.get("username", "Guest"), beta_mode=beta_mode)

@app.route("/privacy")
def privacy():
    return render_template("privacy.html", beta_mode=beta_mode)

@app.route("/terms")
def terms():
    return render_template("terms.html", beta_mode=beta_mode)

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
