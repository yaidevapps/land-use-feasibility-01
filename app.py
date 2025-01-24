import streamlit as st
from PIL import Image
import google.generativeai as genai
from bellevue_land_use_helper import BellevueLandUseFeasibilityAssistant

# Page configuration
st.set_page_config(
    page_title="Bellevue Residential Land Use Feasibility Assistant",
    page_icon="🏘️",
    layout="wide"
)

# Initialize session state
if 'chat' not in st.session_state:
    st.session_state.chat = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'image_analyzed' not in st.session_state:
    st.session_state.image_analyzed = False
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# Sidebar for API key
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        assistant = BellevueLandUseFeasibilityAssistant(api_key)
    else:
        assistant = BellevueLandUseFeasibilityAssistant()
    
    # Add clear chat button
    if st.button("Clear Analysis History"):
        st.session_state.messages = []
        st.session_state.chat = assistant.start_chat()
        st.session_state.image_analyzed = False
        st.rerun()

# Main title
st.title("🏘️ Bellevue Residential Land Use Feasibility Assistant")
st.markdown("""
This AI assistant helps architects and developers assess the feasibility of residential development projects in Bellevue, Washington. 
Discuss zoning requirements, upload site plans, and receive comprehensive land use analysis.
""")

# Initialize chat if not already done
if st.session_state.chat is None:
    st.session_state.chat = assistant.start_chat()

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["💬 Land Use Consultation", "📋 Site Plan Review"])

with tab1:
    st.markdown("### Residential Development Consultation")
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about Bellevue residential zoning requirements..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing Land Use Requirements..."):
                response = assistant.send_message(st.session_state.chat, prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    st.markdown("### Site Plan Feasibility Review")
    # File uploader
    uploaded_file = st.file_uploader("Upload site plans, survey documents, or property images", type=['png', 'jpg', 'jpeg', 'pdf'])

    if uploaded_file:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Site Plan", use_container_width=True)
        
        # Analyze button
        if not st.session_state.image_analyzed:
            if st.button("Perform Land Use Feasibility Analysis", type="primary"):
                with st.spinner("Conducting comprehensive site feasibility review..."):
                    # Store image for reference
                    st.session_state.current_image = image
                    
                    # Get initial analysis
                    report = assistant.analyze_image(image, st.session_state.chat)
                    
                    # Add the report to chat history
                    st.session_state.messages.append({"role": "assistant", "content": report})
                    st.session_state.image_analyzed = True
                    st.rerun()

# Footer with instructions
st.markdown("---")
st.markdown("""
### How to Use the Bellevue Land Use Feasibility Assistant
1. Chat about zoning and development requirements in the Consultation tab
2. For detailed site analysis:
   - Switch to the Site Plan Review tab
   - Upload site plans, survey documents, or property images
   - Click "Perform Land Use Feasibility Analysis" for a comprehensive review
3. Receive detailed reports highlighting development potential and constraints

Example questions:
- What are the zoning requirements for R-4 districts?
- How do setback requirements work in Bellevue?
- What environmental considerations affect residential development?
- Can you explain lot coverage limits?
- What are the height restrictions for residential buildings?
""")

# Download button for analysis history
if st.session_state.messages:
    analysis_history = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages])
    st.download_button(
        "Download Land Use Feasibility Report",
        analysis_history,
        file_name="bellevue_land_use_analysis.txt",
        mime="text/plain"
    )