import os
import sys

import streamlit as st

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

# Check if model exists
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "medisum-llama3-8b")
MODEL_EXISTS = os.path.exists(MODEL_PATH)

# Try to import inference module
MODEL_AVAILABLE = False
IMPORT_ERROR = None
try:
    from inference import generate_soap_note, load_model

    MODEL_AVAILABLE = True
except Exception as e:
    IMPORT_ERROR = str(e)

    # Fallback: Define enhanced mock function for demo mode
    def generate_soap_note(dialogue, use_model=False):
        """Enhanced mock SOAP note generation with better parsing"""
        import re

        dialogue_lower = dialogue.lower()
        lines = dialogue.split("\n")

        # Extract patient symptoms and complaints
        patient_lines = [
            line for line in lines if "patient" in line.lower() or "pt" in line.lower()
        ]
        doctor_lines = [
            line for line in lines if "doctor" in line.lower() or "dr" in line.lower()
        ]

        # Parse subjective information
        subjective_parts = []
        for line in patient_lines[:3]:  # First 3 patient statements
            # Remove "Patient:" or "Pt:" prefix
            text = re.sub(r"^(patient|pt):\s*", "", line, flags=re.IGNORECASE).strip()
            if text:
                subjective_parts.append(text)

        # Parse objective information from doctor's observations
        objective_parts = []
        for line in doctor_lines:
            text = line.lower()
            # Look for clinical observations
            if any(
                word in text
                for word in [
                    "examine",
                    "check",
                    "look",
                    "see",
                    "feel",
                    "hear",
                    "blood pressure",
                    "temperature",
                    "vitals",
                ]
            ):
                obs = re.sub(r"^(doctor|dr):\s*", "", line, flags=re.IGNORECASE).strip()
                if obs and not obs.endswith("?"):  # Skip questions
                    objective_parts.append(obs)

        # Parse assessment (diagnosis keywords)
        assessment = "Clinical evaluation pending"
        assessment_keywords = [
            "appears",
            "looks like",
            "seems",
            "diagnosed",
            "condition",
            "probably",
            "likely",
        ]
        for line in doctor_lines:
            text = line.lower()
            if any(kw in text for kw in assessment_keywords):
                # Extract the diagnosis part
                for kw in assessment_keywords:
                    if kw in text:
                        parts = text.split(kw)
                        if len(parts) > 1:
                            assessment = parts[1].strip().rstrip(".").capitalize()
                            break

        # Parse plan (treatment recommendations)
        plan_parts = []
        plan_keywords = [
            "prescribe",
            "recommend",
            "try",
            "take",
            "should",
            "therapy",
            "follow up",
            "come back",
            "rest",
        ]
        for line in doctor_lines:
            text = line.lower()
            if any(kw in text for kw in plan_keywords):
                plan_text = re.sub(
                    r"^(doctor|dr):\s*", "", line, flags=re.IGNORECASE
                ).strip()
                if plan_text and not plan_text.endswith("?"):
                    plan_parts.append(plan_text)

        # Build SOAP note
        subjective = (
            "Patient reports " + ". ".join(subjective_parts[:2])
            if subjective_parts
            else "Patient presents with symptoms"
        )
        objective = (
            ". ".join(objective_parts[:2])
            if objective_parts
            else "Physical examination performed"
        )
        plan = (
            ". ".join(plan_parts[:2])
            if plan_parts
            else "Treatment plan discussed with patient"
        )

        return f"""**Subjective:** {subjective}

**Objective:** {objective}

**Assessment:** {assessment}

**Plan:** {plan}"""

    def load_model():
        """Placeholder for demo mode"""
        pass


# Page configuration
st.set_page_config(
    page_title="MediSum AI - Medical Documentation Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .soap-output {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #145a8a;
    }
    .example-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown('<h1 class="main-header">🏥 MediSum AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Medical Documentation Assistant - Convert doctor-patient dialogues into structured SOAP notes</p>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        """
        **MediSum AI** uses a fine-tuned **Llama-3-8B** model with QLoRA 
        to automatically generate structured clinical documentation.
        
        **SOAP Format:**
        - **S**ubjective: Patient's symptoms
        - **O**bjective: Clinical observations
        - **A**ssessment: Diagnosis
        - **P**lan: Treatment recommendations
        """
    )

    st.header("⚙️ Model Info")
    st.write("**Base Model:** Meta-Llama-3-8B")
    st.write("**Fine-tuning:** QLoRA (4-bit)")
    st.write("**Dataset:** Medical SOAP notes")

    st.header("📊 Model Status")
    if MODEL_EXISTS:
        st.info(f"📁 Model found at: `models/medisum-llama3-8b`")
    else:
        st.error("❌ Model not found in `models/` folder")

    if MODEL_AVAILABLE:
        st.success("✅ Real model loaded successfully")
        st.caption("Using fine-tuned Llama-3-8B")
    else:
        st.warning("⚠️ Demo mode (Model not loaded)")
        if IMPORT_ERROR:
            with st.expander("🔍 Error Details"):
                st.code(IMPORT_ERROR, language="python")
        st.caption("Install CPU PyTorch to load model:")
        st.code(
            "pip install torch --index-url https://download.pytorch.org/whl/cpu",
            language="bash",
        )

    st.markdown("---")
    st.caption("Built with Streamlit • Powered by Llama-3")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Dialogue")

    # Example dialogues
    st.markdown("**Quick Examples:**")
    examples = {
        "Headache": "Doctor: What brings you in today?\nPatient: I've been having severe headaches for the past three days. They're mostly in the front of my head.\nDoctor: Any other symptoms like nausea or sensitivity to light?\nPatient: Yes, bright lights make it worse, and I feel a bit nauseous.\nDoctor: I see. Let me check your vitals. Your blood pressure is slightly elevated. It appears to be a tension headache, possibly triggered by stress.\nPatient: What should I do?\nDoctor: I'll prescribe some pain medication and recommend rest. Try to avoid screens and get adequate sleep.",
        "Back Pain": "Doctor: Hello, what seems to be the problem?\nPatient: I've had sharp lower back pain for about a week. It gets worse when I sit for too long.\nDoctor: Have you had any numbness or tingling in your legs?\nPatient: Sometimes my right leg feels tingly.\nDoctor: Let me examine your back. I can feel some tenderness in the lumbar region.\nPatient: Is it serious?\nDoctor: It looks like a muscle strain with possible nerve irritation. I'll prescribe ibuprofen and refer you to physical therapy.",
        "Cold Symptoms": "Doctor: How can I help you today?\nPatient: I've had a persistent cough and sore throat since Monday.\nDoctor: Any fever or difficulty breathing?\nPatient: I had a low-grade fever yesterday, around 100 degrees. No trouble breathing, just very tired.\nDoctor: Let me check your throat. It's quite red. Your lungs sound clear though.\nPatient: Is it the flu?\nDoctor: It appears to be a viral upper respiratory infection. Get plenty of rest and fluids. Come back if symptoms worsen.",
    }

    selected_example = st.selectbox(
        "Choose an example:", ["Custom", "Headache", "Back Pain", "Cold Symptoms"]
    )

    if selected_example != "Custom":
        default_text = examples[selected_example]
    else:
        default_text = ""

    dialogue_input = st.text_area(
        "Enter the doctor-patient dialogue:",
        value=default_text,
        height=300,
        placeholder="Doctor: What brings you in today?\nPatient: I've been having chest pain...",
        help="Paste or type the conversation between doctor and patient",
    )

    # Generate button
    generate_btn = st.button(
        "🔄 Generate SOAP Note", type="primary", use_container_width=True
    )

with col2:
    st.subheader("📋 Generated SOAP Note")

    # Output container
    soap_output = st.empty()

    if generate_btn:
        if not dialogue_input.strip():
            st.warning("⚠️ Please enter a dialogue first.")
        else:
            with st.spinner("🔄 Analyzing dialogue and generating SOAP note..."):
                try:
                    # Generate SOAP note
                    if MODEL_AVAILABLE:
                        soap_note = generate_soap_note(dialogue_input, use_model=True)
                    else:
                        # Demo mode
                        soap_note = generate_soap_note(dialogue_input, use_model=False)

                    # Display result
                    with soap_output.container():
                        st.markdown('<div class="soap-output">', unsafe_allow_html=True)
                        st.markdown(soap_note)
                        st.markdown("</div>", unsafe_allow_html=True)

                        # Download button
                        st.download_button(
                            label="📥 Download SOAP Note",
                            data=soap_note,
                            file_name="soap_note.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )

                        # Copy to clipboard
                        if st.button("📋 Copy to Clipboard", use_container_width=True):
                            st.code(soap_note, language=None)
                            st.success("✅ SOAP note ready to copy!")

                except Exception as e:
                    st.error(f"❌ Error generating SOAP note: {str(e)}")
                    st.info(
                        "💡 Make sure the model is properly loaded. Check the sidebar for model status."
                    )
    else:
        with soap_output.container():
            st.info("👆 Enter a dialogue and click 'Generate SOAP Note' to get started")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for demonstration purposes only. 
        Always consult qualified healthcare professionals for medical decisions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model in background (if available)
if MODEL_AVAILABLE and "model_loaded" not in st.session_state:
    with st.spinner("Loading model... (this may take a few minutes on first run)"):
        try:
            load_model()
            st.session_state.model_loaded = True
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
            st.session_state.model_loaded = False
