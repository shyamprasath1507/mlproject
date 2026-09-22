import streamlit as st
import requests

# Page Configuration
st.set_page_config(
    page_title="ECO Impact Analyzer",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a bit more "wow" factor
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("⚙️ AI Engineering Change Impact Analyzer")
st.markdown("Evaluate the potential risk and cost impact of proposed Engineering Change Orders (ECOs) using machine learning.")
st.markdown("---")

# Layout
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("ECO Input Parameters")
    
    with st.form("eco_form"):
        complexity_score = st.slider(
            "Complexity Score", 
            min_value=1.0, 
            max_value=10.0, 
            value=5.0, 
            step=0.5,
            help="1.0 = Trivial change, 10.0 = Complete system overhaul"
        )
        
        department_count = st.number_input(
            "Departments Involved", 
            min_value=1, 
            max_value=5, 
            value=2,
            help="Number of engineering/business departments required to approve and implement the change."
        )
        
        original_cost_usd = st.number_input(
            "Original Estimated Cost (USD)", 
            min_value=0.0, 
            value=50000.0, 
            step=1000.0,
            format="%.2f",
            help="The base estimated cost of the component or system before the change."
        )
        
        change_domain = st.selectbox(
            "Change Domain", 
            options=["Mechanical", "Electrical", "Software"],
            help="The primary engineering domain of the proposed change."
        )
        
        submit_button = st.form_submit_button("Analyze Impact 🚀")

with col2:
    st.subheader("Impact Analysis Results")
    
    if submit_button:
        # Prepare payload
        payload = {
            "complexity_score": complexity_score,
            "department_count": department_count,
            "original_cost_usd": original_cost_usd,
            "change_domain": change_domain
        }
        
        with st.spinner("Analyzing impact using AI models..."):
            try:
                # Make request to backend
                # Note: Uses localhost:8000 when running locally, or docker service name in compose
                # Here we default to localhost assuming they are exposed or running together.
                # In Docker, we might need a different URL if Streamlit reaches out from container vs browser.
                # Since Streamlit makes requests from the server side, it can use http://localhost:8000 
                # if they are in the SAME container (which they are, per the requirements for entrypoint.sh).
                response = requests.post("https://mlproject-zae2.onrender.com/predict", json=payload)
                response.raise_for_status()
                
                result = response.json()
                
                risk = result.get("predicted_risk", "Unknown")
                cost = result.get("predicted_cost_overrun", 0.0)
                components = result.get("downstream_affected_components", [])
                
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                
                # Risk Banner
                if risk == "Low":
                    st.success(f"**Predicted Risk Level: {risk}** ✅")
                elif risk == "Medium":
                    st.warning(f"**Predicted Risk Level: {risk}** ⚠️")
                else:
                    st.error(f"**Predicted Risk Level: {risk}** 🚨")
                
                st.markdown("---")
                
                # Cost Metric
                st.metric(
                    label="Predicted Cost Overrun (USD)", 
                    value=f"${cost:,.2f}",
                    delta=f"{(cost/original_cost_usd)*100:.1f}% of original" if original_cost_usd > 0 else None,
                    delta_color="inverse" # higher is worse
                )
                
                st.markdown("---")
                
                # Downstream Components
                st.subheader("Potential Downstream Affected Components")
                if components:
                    for comp in components:
                        st.markdown(f"- 🧩 {comp}")
                else:
                    st.info("No major downstream components predicted to be affected.")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except requests.exceptions.ConnectionError:
                st.error("Error: Could not connect to the backend server. Is FastAPI running on port 8000?")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("👈 Enter the parameters and click 'Analyze Impact' to see the predictions.")
