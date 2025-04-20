import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from streamlit_lottie import st_lottie
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None
# zayd ben
# Page configuration
st.set_page_config(
    page_title="HealthScribe AI",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check dependencies
missing_deps = []
for dep in ['pandas', 'numpy', 'sklearn', 'joblib', 'streamlit_lottie', 'requests', 'fuzzywuzzy']:
    try:
        __import__(dep)
    except ImportError:
        missing_deps.append(dep)

if missing_deps:
    st.error(f"Missing dependencies: {', '.join(missing_deps)}. Install them using pip install {' '.join(missing_deps)}.")
    st.stop()

# Load Lottie animation
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        st.warning(f"Failed to load animation: {str(e)}")
        return None

lottie_health = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_5tkzkblw.json")

# Email notification function 
def send_email_notification(recipient, risk_level, recommendations):
    if not email_enabled:
        return
    
    # Validate email inputs
    if not all([sender_email, sender_password, recipient_email, smtp_server]):
        st.sidebar.warning("Please fill all email notification fields in the sidebar")
        return
    
    subject = f"🚨 Health Alert: {risk_level} Diabetes Risk Detected"
    
    body = f"""
    <html>
        <body>
            <h2>HealthScribe AI Alert</h2>
            <p>Our assessment indicates you have a <strong>{risk_level}</strong> risk of diabetes.</p>
            <h3>Recommended Actions:</h3>
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in recommendations)}
            </ul>
            <p>Please consult with a healthcare provider as soon as possible.</p>
            <p><em>This is an automated message. For medical emergencies, contact local emergency services.</em></p>
        </body>
    </html>
    """
    
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:  # Changé de SMTP()
                server.login(sender_email, sender_password)
        st.sidebar.success("Notification email sent successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to send email: {str(e)}")

# Improved answer finding function
def find_best_answer(question, df):
    if df is None or df.empty:
        return None, 0, None
    
    # Check if fuzzywuzzy is available
    if fuzz is None:
        # Fallback to simple string matching if fuzzywuzzy not available
        question_lower = question.lower()
        for index, row in df.iterrows():
            if isinstance(row['question'], str) and isinstance(row['answer'], str):
                if question_lower in row['question'].lower():
                    return row['answer'], 100, row['question']
        return None, 0, None
    
    # Use fuzzy matching if available
    best_score = 0
    best_answer = None
    best_question = None
    for index, row in df.iterrows():
        if isinstance(row['question'], str) and isinstance(row['answer'], str):
            score = fuzz.partial_ratio(question.lower(), row['question'].lower())
            if score > best_score and score > 70:
                best_score = score
                best_answer = row['answer']
                best_question = row['question']
    return best_answer, best_score, best_question

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary: #111112;
        --secondary: #3a0ca3;
        --accent: #4895ef;
        --success: #38b000;
        --danger: #d00000;
        --warning: #ffaa00;
        --light: #f8f9fa;
        --dark: #212529;
        --text-primary: #2b2d42;
        --text-secondary: #4a4e69;
    }
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: var(--text-primary);
    }
    
    .main {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Improved cards with better contrast */
    .prediction-card, .chatbot-card {
        border-radius: 16px;
        padding: 28px;
        margin-top: 30px;
        background: white;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        color: var(--text-primary) !important;
    }
    
    .positive {
    border-left: 6px solid var(--danger);
    background: linear-gradient(to right, rgba(208, 0, 0, 0.3), #111) !important;
    color: white !important;
}

    /* Ensure all text elements are visible on dark background */
    .positive h3,
    .positive p,
    .positive strong,
    .positive ul,
    .positive li {
        color: white !important;
    }
    .positive h3 {
        color: var(--danger) !important;
    }
    
    .negative {
        border-left: 6px solid var(--success);
        background: linear-gradient(to right, rgba(56, 176, 0, 0.3), #111) !important;
        color: white !important;
    }

    /* Ensure text visibility */
    .negative h3,
    .negative p,
    .negative strong,
    .negative ul,
    .negative li {
        color: white !important;
    }
    
    .negative h3 {
        color: white !important;
    }
    
    .chatbot-response {
    border-left: 6px solid var(--primary);
    background: linear-gradient(to right, rgba(0, 0, 0, 0.7), black) !important;
    color: white !important; /* Ensure text is readable on dark background */
}

    /* Ensure all text elements inside chatbot response are visible */
    .chatbot-response h3,
    .chatbot-response p,    
    .chatbot-response strong,
    .chatbot-response ul,
    .chatbot-response li {
        color: white !important;
    }
    
    .chatbot-response h3 {
        color: white !important;
    }
    
    /* Ensure text is readable in all cards */
    .prediction-card p, .prediction-card ul, 
    .chatbot-card p, .chatbot-card ul {
        color: white !important;
    }
    
    .prediction-card strong, .chatbot-card strong {
        color: white !important;
    }
    
    /* Footer text contrast */
    .footer-note {
        color: var(--text-secondary) !important;
    }
    
    /* Rest of your existing CSS remains the same */
    .stButton>button {
        background: linear-gradient(to right, var(--primary), var(--secondary));
        color: white;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 16px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(67, 97, 238, 0.3);
    }
    
    .prediction-card:hover, .chatbot-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.1);
    }
    
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e9ecef;
        padding: 15px;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 2px rgba(72, 149, 239, 0.2);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 8px 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 24px;
        background: #e9ecef;
        border-radius: 12px;
        font-weight: 600;
        color: var(--text-secondary);
        transition: all 0.3s ease;
        margin: 0 4px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(to right, var(--primary), var(--secondary));
        color: white;
        box-shadow: 0 4px 12px rgba(67, 97, 238, 0.2);
    }
    
    .stSlider .st-ae {
        color: var(--primary) !important;
    }
    
    .stSlider .st-af {
        background-color: var(--primary) !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text-primary);
    }
    
    .stMarkdown p {
        color: var(--text-secondary);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .prediction-card, .chatbot-card {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Header with animation
col1, col2 = st.columns([3, 1])
with col1:
    st.title("💙 HealthScribe AI")
    st.markdown(""" 
    <div style="color: #5f6c7b; font-size: 18px; margin-bottom: 30px;">
        Transform Medical Chaos into Clarity Instantly.
    </div>
    """, unsafe_allow_html=True)
with col2:
    if lottie_health:
        st_lottie(lottie_health, height=150, key="health")

# Sidebar
with st.sidebar:
    st.subheader("Notification Settings")
    email_enabled = st.checkbox("Enable Email Notifications")
    if email_enabled:
        sender_email = st.text_input("Sender Email")
        sender_password = st.text_input("Sender Password", type="password")
        recipient_email = st.text_input("Recipient Email")
        smtp_server = st.text_input("SMTP Server", "smtp.gmail.com")
        smtp_port = st.number_input("SMTP Port", 465)
    st.markdown("""
    <div style="padding: 20px;">
        <h2>About This WebSite</h2>
        <p>Assess your health risks and get trusted medical insights instantly.</p>
        <p style="font-size: 12px; margin-top: 30px;">
        <i class="fas fa-exclamation-circle"></i> For educational purposes only. Not a substitute for medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Load medical dataset
@st.cache_data
def load_medical_data():
    try:
        df = pd.read_csv('medquad.csv')
        if df.empty:
            st.error("Medical dataset is empty.")
            return None
        return df
    except FileNotFoundError:
        st.error("Medical dataset (medquad.csv) not found in the current directory.")
        return None
    except Exception as e:
        st.error(f"Error loading medical dataset: {str(e)}")
        return None

med_data = load_medical_data()

# Load or train model
@st.cache_resource
def get_model():
    try:
        model = joblib.load('diabetes_model.pkl')
        st.sidebar.success("Model loaded successfully")
        return model
    except FileNotFoundError:
        st.sidebar.info("Training new model...")
    dataset_path = 'health_care_diabetes.csv'
    if not os.path.exists(dataset_path):
        st.sidebar.error("Diabetes dataset not found")
        return None
    try:
        df = pd.read_csv(dataset_path)
        zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df[zero_features] = df[zero_features].replace(0, np.nan)
        df.fillna(df.median(), inplace=True)
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, 'diabetes_model.pkl')
        st.sidebar.success("Model trained successfully")
        return model
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")
        return None

model = get_model()

if model is None:
    st.error("⚠️ Could not load the prediction model. Ensure 'health_care_diabetes.csv' is in the directory.")
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["🔍 Diabetes Risk Assessment", "💬 Medical Chatbot"])

# Diabetes Risk Assessment Tab
with tab1:
    st.subheader("Diabetes Risk Assessment")
    st.markdown(""" 
    <div style="color: #5f6c7b; font-size: 16px; margin-bottom: 30px;">
        Fill in your health metrics to assess your diabetes risk.
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        cols = st.columns(2)
        with cols[0]:
            pregnancies = st.slider("Number of Pregnancies", 0, 20, 0)
            age = st.slider("Your Age", 1, 120, 25)
            glucose = st.slider("Glucose Level", 0, 300, 100)
            blood_pressure = st.slider("Blood Pressure", 0, 150, 70)
        with cols[1]:
            skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
            insulin = st.slider("Insulin Level", 0, 1000, 80)
            bmi = st.slider("Body Mass Index", 0.0, 70.0, 25.0, 0.1)
            diabetes_pedigree = st.slider("Diabetes Pedigree", 0.0, 3.0, 0.5, 0.001)
        submitted = st.form_submit_button("🔍 Assess My Risk")

    # Prediction results
    if submitted:
        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                  insulin, bmi, diabetes_pedigree, age]],
                                columns=['Pregnancies', 'Glucose', 'BloodPressure', 
                                        'SkinThickness', 'Insulin', 'BMI', 
                                        'DiabetesPedigreeFunction', 'Age'])
        try:
            with st.spinner('Analyzing your health profile...'):
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1]
                st.markdown("---")
                st.subheader("📋 Your Diabetes Risk Assessment")
                if prediction == 1:
                    recommendations = [
                    "Schedule an appointment with your doctor",
                    "Monitor blood sugar levels regularly",
                    "Follow a balanced diet low in processed sugars",
                    "Engage in regular physical activity",
                    "Attend diabetes education classes if available"
                    ]
    
                    st.markdown(f"""
                    <div class="prediction-card positive">
                        <h3 style="color: #FF6347;">🛑 Higher Risk Detected</h3>
                        <p>Your probability of diabetes is: <strong>{probability:.1%}</strong></p>
                        <p>Recommendations:</p>
                        <ul>
                            {''.join(f'<li>{rec}</li>' for rec in recommendations)}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
    
                    if email_enabled and recipient_email and all([sender_email, sender_password, smtp_server]):
                        send_email_notification(
                        recipient_email,
                        "High",
                        recommendations
                        )
                else:
                    st.markdown(f"""
                    <div class="prediction-card negative">
                        <h3 style="color: #4CAF50;">✅ Lower Risk Detected</h3>
                        <p>Your probability of diabetes is: <strong>{probability:.1%}</strong></p>
                        <p>To maintain health:</p>
                        <ul>
                            <li>Continue healthy eating</li>
                            <li>Stay active</li>
                            <li>Regular check-ups</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

# Medical Chatbot Tab
with tab2:
    st.subheader("Medical Information Chatbot")
    st.markdown("""
    <div style="color: #5f6c7b; font-size: 16px; margin-bottom: 30px;">
        Ask about medical conditions like Glaucoma or High Blood Pressure.
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("chatbot_form"):
        user_question = st.text_area("Type your medical question:", 
                                   placeholder="E.g., What are the symptoms of Glaucoma?",
                                   height=150)
        chatbot_submitted = st.form_submit_button("🤖 Get Answer")
    
    if chatbot_submitted and user_question:
        with st.spinner("Dr. AI is thinking..."):
            answer, score, matched_question = find_best_answer(user_question, med_data)
            if answer and score > 70:
                st.markdown(f"""
                <div class="chatbot-card chatbot-response">
                    <h3 style="color: #4776E6;">Dr. AI's Response</h3>
                    <p><strong>Regarding:</strong> {matched_question}</p>
                    <p>{answer}</p>
                    <p style="color: #7f8c8d;">Consult a healthcare provider for medical advice.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chatbot-card">
                    <h3 style="color: #FF4B4B;">No Answer Found</h3>
                    <p>Sorry, I couldn't find an answer. Try rephrasing or ask about another condition.</p>
                    <p style="color: #7f8c8d;">Consult a healthcare provider for medical advice.</p>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 14px; padding: 20px;">
    <p>For informational purposes only. Consult a qualified healthcare provider for medical concerns.</p>
</div>
""", unsafe_allow_html=True)