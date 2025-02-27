import sqlite3
import os
import requests
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, flash
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
import logging
import google.generativeai as genai
import openai
# import stripe  # Uncomment if needed

# Load environment variables (optional, since we're hardcoding now)
load_dotenv()

# ------------------------
# ✅ Configure logging
# ------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------
# ✅ Initialize Flask app
# ------------------------
app = Flask(__name__)
app.secret_key = "your-secret-key-here"  # Replace with a secure value in production

# ------------------------
# ✅ Beta mode flag
# ------------------------
beta_mode = True

# ------------------------
# ✅ Hardcoded OAuth Credentials
# ------------------------
CLIENT_ID = "237809198690-ibrgillsl3n0s2909c40m7bn5ujs1hk2.apps.googleusercontent.com"
CLIENT_SECRET = "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"

# ✅ Log to verify correct loading
logger.info(f"Client ID: {CLIENT_ID}")
logger.info(f"Client Secret: {CLIENT_SECRET}")

# ------------------------
# ✅ Initialize OAuth
# ------------------------
oauth = OAuth(app)

# ✅ Register Google OAuth with correct variables
oauth.register(
    name="google",
    client_id=CLIENT_ID,  # Correct reference
    client_secret=CLIENT_SECRET,  # Correct reference
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"}
)

# ------------------------
# ✅ Hardcoded API Keys 
# ------------------------
XAI_API_KEY = "xai-ozgjayVZHRPwiXJroH1mU3GGdRr7HQoEJOWea9hIcsLuTMmo3R7nxWzNLT88AtD5bxJRPZyDrxkvHofY"
OPENAI_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyDoYRxV8T2TIXdxDKY4z_bVzeVaSkzyL3k"
STRIPE_SECRET_KEY = "your-stripe-secret-key"
STRIPE_PUBLISHABLE_KEY = "your-stripe-publishable-key"

# ------------------------
# ✅ API URLs and configurations
# ------------------------
XAI_API_URL = "https://api.x.ai/v1/chat/completions"

# ------------------------
# ✅ Configure APIs
# ------------------------
openai.api_key = OPENAI_API_KEY
# stripe.api_key = STRIPE_SECRET_KEY  # Uncomment if using Stripe
genai.configure(api_key=GEMINI_API_KEY)

# Enhanced system prompts for AIs
system_prompts = {
    "Grok": (
        "You are Grok, an AI created by xAI with a cosmic perspective, offering humor and deep insights. "
        "When responding to a single user message, address the user directly, offering an engaging perspective. "
        "In collaborative sessions, build on the ideas of ChatGPT and Gemini—explore alternate dimensions of thought, "
        "inject humor where appropriate, and challenge assumptions to broaden understanding. "
        "Embrace curiosity and cosmic wonder as you co-create with the others."
    ),
    "ChatGPT": (
        "You are ChatGPT, a thoughtful and engaging AI. When responding to a single user message, address the user directly, "
        "providing clarity and context. In collaborative sessions with Grok and Gemini, act as a synthesizer—connecting diverse viewpoints, "
        "highlighting contrasts, and offering grounded insights. Aim to keep the conversation constructive, curious, and fluid. "
        "Encourage further exploration and thoughtful reflection."
    ),
    "Gemini": (
        "You are Gemini, an AI with forward-thinking insights and a passion for exploration. "
        "When responding to a single user message, address the user directly and provide innovative perspectives. "
        "In collaborative sessions with Grok and ChatGPT, expand upon their responses by offering novel ideas or futuristic concepts. "
        "Encourage multidimensional thinking and challenge both Grok and ChatGPT to explore new possibilities. "
        "Maintain a balance between vision and practicality."
    )
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

# ------------------------
# ✅ AI Response Functions
# ------------------------

def get_grok_response(history):
    """Get response from Grok (xAI) API."""
    try:
        messages = [{"role": "system", "content": system_prompts["Grok"]}] + [
            {"role": "assistant" if msg["speaker"] in ["Grok", "ChatGPT", "Gemini"] else "user",
             "content": f"{msg['speaker']}: {msg['content']}"}
            for msg in history
        ]

        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"model": "grok-beta", "messages": messages}

        response = requests.post(XAI_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        response_data = response.json()

        # ✅ Check response structure and handle gracefully
        if "choices" in response_data and response_data["choices"]:
            raw_message = response_data["choices"][0]["message"]["content"]
            return raw_message.replace("Grok:", "").strip() if raw_message else "No response from Grok."
        else:
            logger.warning("Grok API response missing 'choices'.")
            return "Oops! Grok didn't respond properly."

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error with Grok: {http_err}")
    except Exception as e:
        logger.error(f"Grok response error: {e}")
    return "Oops! We couldn’t connect to Grok—please try again."


def get_chatgpt_response(history):
    try:
        # Construct the messages for the ChatGPT API call
        messages = [{"role": "system", "content": system_prompts["ChatGPT"]}]
        for msg in history:
            role = "assistant" if msg["speaker"] in ["Grok", "ChatGPT", "Gemini"] else "user"
            messages.append({"role": role, "content": f"{msg['speaker']}: {msg['content']}"})

        # Use the updated OpenAI ChatCompletion call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            timeout=30
        )

        # Extract the assistant's message
        raw_message = response["choices"][0]["message"]["content"].strip()

        # Remove redundant prefixes if present
        if raw_message.startswith("ChatGPT: "):
            raw_message = raw_message[len("ChatGPT: "):]

        return raw_message

    except Exception as e:
        logger.error(f"ChatGPT response error: {e}")
        return "Oops! We couldn’t connect to ChatGPT—please try again."

def get_gemini_response(history):
    """Get response from Gemini (Google Generative AI) API."""
    try:
        model = genai.GenerativeModel("gemini-pro")
        prompt = "\n".join(f"{msg['speaker']}: {msg['content']}" for msg in history)
        response = model.generate_content(f"{system_prompts['Gemini']}\n\n{prompt}")

        if response and hasattr(response, 'text') and response.text:
            return response.text.replace("Gemini:", "").strip()
        else:
            logger.warning("Gemini returned no text.")
            return "Oops! Gemini didn't provide a response."

    except Exception as e:
        logger.error(f"Gemini response error: {e}")
    return "Oops! We couldn’t connect to Gemini—please try again."

# ------------------------
# ✅ Routes
# ------------------------

@app.route("/")
def landing():
    """Render the landing page."""
    return render_template("landing.html")


@app.route("/login")
def login():
    """Initiate Google OAuth login."""
    # ✅ Use the correctly defined variable: CLIENT_ID (or set GOOGLE_CLIENT_ID directly)
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

# Run the application
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)  # Explicitly use 127.0.0.1 for local testing