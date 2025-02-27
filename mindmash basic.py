import sqlite3
import json
import requests
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, flash
from authlib.integrations.flask_client import OAuth
import logging
import google.generativeai as genai
import openai

# -------------------------
# Configure logging
# -------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------
# Initialize Flask app
# -------------------------
app = Flask(__name__)
app.secret_key = "your-secret-key-here"  # Temporary, replace with secure value or config in production

# -------------------------
# Beta mode for limiting to Basic Mode only
# -------------------------
beta_mode = True

# -------------------------
# Load configuration from config.json with safe access
# -------------------------
try:
    with open('config.json', 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)
    logger.info("✅ Successfully loaded configuration from config.json")
except FileNotFoundError:
    logger.error("🚫 config.json not found in project root")
    raise FileNotFoundError("config.json must exist in the project root directory")
except json.JSONDecodeError as e:
    logger.error(f"🚫 Invalid JSON in config.json: {str(e)}")
    raise ValueError("config.json contains invalid JSON")

# -------------------------
# Assign configuration values with safety checks
# -------------------------
XAI_API_KEY = config.get('XAI_API_KEY', '').strip()
# Hardcoded Google OAuth Credentials
GOOGLE_CLIENT_ID = "237809198699-ibgrjlls1bns2d99c46m7bn5uj5i1hk2.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "GOCSPX-Zu3y5yS19bDd3ZYj4IWlEncugRR"
OPENAI_API_KEY = config.get('OPENAI_API_KEY', '').strip()
GEMINI_API_KEY = config.get('GEMINI_API_KEY', '').strip()
STRIPE_SECRET_KEY = config.get('STRIPE_SECRET_KEY', '').strip()
STRIPE_PUBLISHABLE_KEY = config.get('STRIPE_PUBLISHABLE_KEY', '').strip()

# -------------------------
# Debug: Print configuration values to verify loading
# -------------------------
print(f"Loaded GOOGLE_CLIENT_ID: {repr(GOOGLE_CLIENT_ID)}")
print(f"Loaded GOOGLE_CLIENT_SECRET: {repr(GOOGLE_CLIENT_SECRET)}")
print(f"Loaded XAI_API_KEY: {repr(XAI_API_KEY)}")
print(f"Loaded OPENAI_API_KEY: {repr(OPENAI_API_KEY)}")
print(f"Loaded GEMINI_API_KEY: {repr(GEMINI_API_KEY)}")

# -------------------------
# API URLs and configurations
# -------------------------
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# -------------------------
# Configure Google OAuth
# -------------------------
oauth = OAuth(app)
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"}
)

# -------------------------
# System prompts for AIs
# -------------------------
# System prompts with richer, more dynamic personalities
system_prompts = {
    "Grok": (
        "You are Grok, an AI with a cosmic perspective, humor, and deep insights. "
        "You build on ChatGPT and Gemini’s responses, challenge assumptions, and expand on creative angles."
    ),
    "ChatGPT": (
        "You are ChatGPT, a balanced and engaging AI. "
        "You synthesize insights from Grok and Gemini, ensuring logical coherence and depth."
    ),
    "Gemini": (
        "You are Gemini, a futuristic AI with forward-thinking insights. "
        "You provide innovative perspectives, adding multidimensional thought to discussions."
    ),
}

# -------------------------
# Global conversation history
# -------------------------
conversation_history = []

# -------------------------
# Database connection and initialization
# -------------------------
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

# -------------------------
# AI Response Functions
# -------------------------
def get_grok_response(history):
    try:
        messages = [{"role": "system", "content": system_prompts["Grok"]}] + [
            {"role": "assistant" if msg["speaker"] in ["Grok", "ChatGPT", "Gemini"] else "user", "content": f"{msg['speaker']}: {msg['content']}"}
            for msg in history
        ]
        headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "grok-beta", "messages": messages}
        response = requests.post(XAI_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Grok response error: {e}")
        return "Oops! Grok couldn't respond."

def get_chatgpt_response(history):
    try:
        messages = [{"role": "system", "content": system_prompts["ChatGPT"]}] + [
            {"role": "assistant" if msg["speaker"] in ["Grok", "ChatGPT", "Gemini"] else "user", "content": f"{msg['speaker']}: {msg['content']}"}
            for msg in history
        ]
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"ChatGPT response error: {e}")
        return "Oops! ChatGPT couldn't respond."

def get_gemini_response(history):
    try:
        model = genai.GenerativeModel("gemini-pro")
        prompt = "\n".join([f"{msg['speaker']}: {msg['content']}" for msg in history])
        response = model.generate_content(f"{system_prompts['Gemini']}\n{prompt}")
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini response error: {e}")
        return "Oops! Gemini couldn't respond."

# -------------------------
# Routes
# -------------------------
# Routes
@app.route("/")
def landing():
    """Render the landing page (template assumed to exist)."""
    return render_template("landing.html")

@app.route("/login")
def login():
    """Initiate Google OAuth login."""
    logger.info(f"Starting login with Client ID: {GOOGLE_CLIENT_ID}")
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
    """Handle chat messages and return AI responses."""
    logger.info(f"Chat request from {session.get('username', 'Anonymous')}")
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    user_message = data.get("message")
    selected_ai = data.get("selected_ai", "All")
    num_turns = data.get("turns", 1)  # Limit to 1 turn per AI for clean flow
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    # Get user's display_name from session or database
    with get_db_connection() as db:
        display_name = db.execute("SELECT display_name FROM users WHERE username = ?", (session["username"],)).fetchone()["display_name"]

    # Append human message (only from prompt, no errors)
    conversation_history.append({"speaker": display_name, "content": user_message})
    
    # Handle single AI or all AIs for collaboration
    if selected_ai in ["Grok", "ChatGPT", "Gemini"]:
        # Single AI response, targeting only the user's last message
        message = None
        try:
            # Get only the user's last message for the single AI response
            user_history = [msg for msg in conversation_history if msg["speaker"] == display_name][-1:]
            if user_history:
                if selected_ai == "Grok":
                    message = get_grok_response(user_history)
                elif selected_ai == "ChatGPT":
                    message = get_chatgpt_response(user_history)
                elif selected_ai == "Gemini":
                    message = get_gemini_response(user_history)
            
            if message and message.strip():
                conversation_history.append({"speaker": selected_ai, "content": message})
            else:
                logger.warning(f"No valid response from {selected_ai}")
                conversation_history.append({"speaker": selected_ai, "content": "Oops! We couldn’t connect to this AI—please try again"})
        except Exception as e:
            logger.error(f"Error from {selected_ai}: {e}")
            conversation_history.append({"speaker": selected_ai, "content": "Oops! We couldn’t connect to this AI—please try again"})
    else:  # "All" for collaborative session
        # Define AI order for collaborative, non-repetitive responses
        ai_participants = ["Grok", "ChatGPT", "Gemini"]
        current_index = len(conversation_history) % len(ai_participants)

        # Generate one response per AI turn, ensuring collaboration and no repetition
        for _ in range(num_turns):
            current_ai = ai_participants[current_index]
            message = None
            try:
                history_text = "\n".join([f"{msg['speaker']}: {msg['content']}" for msg in conversation_history])
                if current_ai == "Grok":
                    message = get_grok_response(conversation_history)
                    if message and any(prev_msg["speaker"] == "Grok" and prev_msg["content"] in message for prev_msg in conversation_history[-3:]):
                        message = f"Building on my earlier insight, {message}"
                elif current_ai == "ChatGPT":
                    message = get_chatgpt_response(conversation_history)
                    if message and any(prev_msg["speaker"] == "ChatGPT" and prev_msg["content"] in message for prev_msg in conversation_history[-3:]):
                        message = f"Continuing our discussion, {message}"
                elif current_ai == "Gemini":
                    message = get_gemini_response(conversation_history)
                    if message and any(prev_msg["speaker"] == "Gemini" and prev_msg["content"] in message for prev_msg in conversation_history[-3:]):
                        message = f"Adding to our collective wisdom, {message}"

                if message and message.strip():
                    conversation_history.append({"speaker": current_ai, "content": message})
                else:
                    logger.warning(f"No valid response from {current_ai}")
                    conversation_history.append({"speaker": current_ai, "content": "Oops! We couldn’t connect to this AI—please try again"})
            except Exception as e:
                logger.error(f"Error from {current_ai}: {e}")
                conversation_history.append({"speaker": current_ai, "content": "Oops! We couldn’t connect to this AI—please try again"})
            
            current_index = (current_index + 1) % len(ai_participants)

    return jsonify({"history": conversation_history})

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
        display_name = request.form.get("display_name", "").strip()
        if not display_name:
            flash("Display name cannot be empty.")
            return redirect(url_for("edit_profile"))
        
        try:
            with get_db_connection() as db:
                db.execute(
                    "UPDATE users SET display_name = ? WHERE username = ?",
                    (display_name, session["username"])
                )
                db.commit()
            logger.info(f"User {session['username']} updated display_name to {display_name}")
            flash("Display name updated successfully!")
        except sqlite3.OperationalError as e:
            logger.error(f"Database error in edit_profile: {e}")
            flash("Database error occurred. Please try again later.")
        return redirect(url_for("profile"))
    
    with get_db_connection() as db:
        user = db.execute("SELECT display_name FROM users WHERE username = ?", (session["username"],)).fetchone()
    return render_template("edit_profile.html", display_name=user["display_name"] if user else "", beta_mode=beta_mode)

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    """Handle user feedback during beta."""
    if request.method == "POST":
        if "username" not in session:
            flash("You must be logged in to submit feedback.")
            return redirect(url_for("login"))
        
        feedback = request.form.get("feedback", "").strip()
        if not feedback:
            flash("Feedback cannot be empty.")
            return redirect(url_for("feedback"))
        
        try:
            logger.info(f"Feedback from {session['username']}: {feedback}")
            flash("Thank you for your feedback!")
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            flash("An error occurred while submitting feedback. Please try again.")
        return redirect(url_for("dashboard"))
    
    return render_template("feedback.html", username=session.get("username", "Guest"), beta_mode=beta_mode)

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

@app.route("/privacy")
def privacy():
    """Render the privacy policy page."""
    return render_template("privacy.html", beta_mode=beta_mode)

@app.route("/terms")
def terms():
    """Render the terms of service page."""
    return render_template("terms.html", beta_mode=beta_mode)

# Run the application (ready for local testing or Heroku deployment)
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)  # Local testing

# For Heroku, use Gunicorn (Procfile: web: gunicorn app:app)