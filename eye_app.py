import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from groq import Groq
from fpdf import FPDF
import base64
import datetime
import tempfile
import os

# Set wide layout
st.set_page_config(layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
        .main .block-container {
            padding: 2rem 5rem;
        }
        h1, h2, h3 {
            color: #1f77b4;
        }
        .stTextInput > div > div > input {
            border: 1px solid #ccc;
            border-radius: 0.5rem;
            padding: 0.5rem;
        }
        .stNumberInput > div > div > input {
            border: 1px solid #ccc;
            border-radius: 0.5rem;
            padding: 0.5rem;
        }
        .stSelectbox > div > div > div {
            border: 1px solid #ccc;
            border-radius: 0.5rem;
            padding: 0.5rem;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            border-radius: 0.5rem;
            padding: 0.6rem 1.2rem;
            font-weight: bold;
        }
        .stDownloadButton>button {
            background-color: #28a745;
            color: white;
            border-radius: 0.5rem;
            padding: 0.6rem 1.2rem;
            font-weight: bold;
        }
        
        }
    </style>
""", unsafe_allow_html=True)

# Load model
model = load_model("/Users/sarumathyj/Downloads/eye_disease_model_v2s.h5", compile=False)
model.compile()

# Labels
class_labels = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Healthy']
img_size = (224, 224)

# Groq API key
API_KEY = ""
client = Groq(api_key=API_KEY)

# Treatment info from Groq
def get_treatment_and_risk(disease):
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": f"Give me detailed treatment plan and risk assessment for {disease} in bullet points"}
        ],
        model="llama3-70b-8192"
    )
    if response:
        return response.choices[0].message.content
    else:
        return "Error: Unable to fetch treatment information"

# PDF generation
def create_pdf_report(patient_info, image_path, prediction, confidence, treatment):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "RetinIQ - Eye Disease Diagnosis Report", 0, 1, 'C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Patient Information", 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Name: {patient_info['name']}", 0, 1)
    pdf.cell(0, 10, f"Age: {patient_info['age']}", 0, 1)
    pdf.cell(0, 10, f"Gender: {patient_info['gender']}", 0, 1)
    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Retinal Image:", 0, 1)
    pdf.image(image_path, x=50, w=100)
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Diagnosis Results:", 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Predicted Condition: {prediction}", 0, 1)
    pdf.cell(0, 10, f"Confidence Score: {confidence:.2f}%", 0, 1)
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Treatment Plan & Risk Assessment:", 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, treatment)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = temp_file.name
    pdf.output(pdf_path)

    return pdf_path

# Page title
st.title("👁️ RetinIQ - AI Powered Eye Disease Detector")
st.markdown("Upload a retinal image and enter patient info to receive an AI-generated diagnosis, treatment plan, and downloadable report.")

# Form
with st.form("patient_info"):
    st.subheader("Patient Information")
    name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=1, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg", "png", "jpeg"])
    submitted = st.form_submit_button("Submit")

# Processing submission
if submitted and uploaded_file is not None and name:
    with st.spinner(" Analyzing retinal image and generating report..."):
        image_pil = Image.open(uploaded_file)
        img = image_pil.resize(img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        treatment = get_treatment_and_risk(predicted_class)


    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="report-section">', unsafe_allow_html=True)
        st.subheader("📋 Patient Details")
        st.write(f"**Name:** {name}")
        st.write(f"**Age:** {age}")
        st.write(f"**Gender:** {gender}")
        st.image(image_pil, caption='Retinal Image', width=300)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="report-section">', unsafe_allow_html=True)
        st.subheader("🔍 Diagnosis Results")
        st.markdown(f"**Predicted Condition:** <span style='color:#50C878; font-size:20px;'>{predicted_class}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence Score:** <span style='color:#39FF14; font-size:20px;'>{confidence:.2f}%</span>", unsafe_allow_html=True)

        st.subheader("💊 Treatment Plan & Risk Assessment")
        st.write(treatment)
        st.markdown('</div>', unsafe_allow_html=True)

    patient_info = {
        "name": name,
        "age": age,
        "gender": gender
    }

    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image_pil.save(temp_img.name)

    pdf_path = create_pdf_report(patient_info, temp_img.name, predicted_class, confidence, treatment)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    st.download_button(
        label="📄 Download Full Report (PDF)",
        data=pdf_bytes,
        file_name=f"RetinIQ_Report_{name.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf"
    )

elif submitted and not name:
    st.error("Please enter patient name.")
elif submitted and not uploaded_file:
    st.error("Please upload an eye image.")
