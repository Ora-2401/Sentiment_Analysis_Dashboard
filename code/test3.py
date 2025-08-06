import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import io
import json
import time
import os  # Added missing import
from datetime import datetime
from components.sentiment_analysis import analyze_sentiments_vader, analyze_sentiments_hf
from components.utils import parse_text_input
from components.visuals import export_data

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Streamlit page configuration
st.set_page_config(
    page_title="Sentiment Explorer Dashboard 💬",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced UI with dark theme, animations, and interactivity
st.markdown(
    """
    <style>
    /* General styling */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #1C2526;
        color: #F5F6F5;
    }
    .main .block-container {
        background-color: #1C2526;
        padding: 20px;
        border-radius: 10px;
    }
    .big-font {
        font-size: 34px !important;
        text-align: center;
        color: #F5F6F5;
        margin-bottom: 10px;
        font-weight: 600;
        animation: fadeIn 1s ease-in;
    }
    .subtitle {
        text-align: center;
        color: #7F8C8D;
        font-size: 18px !important;
        margin-bottom: 20px;
        animation: fadeIn 1.2s ease-in;
    }
    .section-header {
        font-size: 22px !important;
        color: #F5F6F5;
        font-weight: 500;
        margin-top: 20px;
        margin-bottom: 10px;
        animation: fadeIn 1s ease-in;
    }
    .info-box {
        background-color: #2E3537;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #00C4B4;
        color: #F5F6F5;
        font-size: 14px;
        animation: fadeIn 1s ease-in;
    }
    .stButton>button {
        background-color: #00C4B4;
        color: #F5F6F5;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #009B8D;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        color: #F5F6F5;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00C4B4 !important;
        color: #F5F6F5 !important;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2E3537;
        color: #F5F6F5;
    }
    .sidebar .sidebar-content {
        background-color: #2E3537;
        padding: 20px;
        border-radius: 8px;
    }
    .stRadio > label, .stSelectbox > label, .stCheckbox > label {
        color: #F5F6F5;
        font-weight: 500;
        font-size: 15px;
    }
    .stTextArea > label, .stFileUploader > label {
        color: #F5F6F5;
        font-weight: 500;
    }
    .stTextArea textarea, .stTextInput input {
        background-color: #2E3537;
        color: #F5F6F5;
        border: 1px solid #00C4B4;
        border-radius: 8px;
    }
    .stSpinner > div > div {
        border-top-color: #00C4B4 !important;
    }
    /* Interactive features */
    .hover-info {
        position: relative;
        display: inline-block;
    }
    .hover-info .tooltip {
        visibility: hidden;
        width: 200px;
        background-color: #2E3537;
        color: #F5F6F5;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .hover-info:hover .tooltip {
        visibility: visible;
        opacity: 1;
    }
    /* Animation keyframes */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
    <h1 class="big-font">Sentiment Explorer Dashboard 💬</h1>
    <p class="subtitle">Analyze the emotional tone of your text data with precision and style.</p>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if 'text_data' not in st.session_state:
    st.session_state.text_data = []
if 'selected_texts' not in st.session_state:
    st.session_state.selected_texts = []
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "VADER (local, fast)"
if 'api_token' not in st.session_state:
    st.session_state.api_token = ""
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.5
if 'extract_keywords' not in st.session_state:
    st.session_state.extract_keywords = True
if 'include_neutral' not in st.session_state:
    st.session_state.include_neutral = True

# Sidebar: Input and Configuration
with st.sidebar:
    st.markdown('<div class="section-header">📥 Text Input</div>', unsafe_allow_html=True)
    input_type = st.selectbox(
        "Data Source Type",
        ["Reviews", "Social Media", "Custom"],
        help="Select the type of text data to analyze."
    )
    user_input = st.text_area(
        "Enter Text (one item per line)",
        height=200,
        placeholder="Type or paste one item per line...",
        help="Enter text data for analysis."
    )
    uploaded_file = st.file_uploader("Upload File (CSV/TXT)", type=["csv", "txt"])
    
    st.markdown('<div class="section-header">🛠️ Analysis Settings</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Model",
        ["VADER (local, fast)", "Hugging Face (API, accurate)", "Compare Both"],
        help="Choose sentiment analysis model.",
        key="model_choice"
    )
    
    if model_choice in ["Hugging Face (API, accurate)", "Compare Both"]:
        api_token = st.text_input(
            "Hugging Face API Token",
            type="password",
            help="Required for Hugging Face model.",
            key="api_token"
        )
    else:
        st.session_state.api_token = ""
    
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        0.0, 1.0, 0.5, 0.01, 
        key="confidence_threshold"
    )
    extract_keywords = st.checkbox(
        "Extract Keywords", 
        value=True, 
        key="extract_keywords"
    )
    include_neutral = st.checkbox(
        "Include Neutral Sentiment", 
        value=True, 
        key="include_neutral"
    )

    if st.button("Analyze", key="analyze_button"):
        with st.spinner("Analyzing..."):
            text_data = parse_text_input(user_input) if user_input else []
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                        text_column = st.selectbox(
                            "Select Text Column", 
                            df.columns, 
                            key="text_column"
                        )
                        text_data.extend(df[text_column].dropna().astype(str).tolist())
                    else:
                        content = uploaded_file.read().decode("utf-8")
                        text_data.extend([line for line in content.splitlines() if line.strip()])
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
            
            st.session_state.text_data = [text for text in text_data if text.strip()]
            
            # Perform analysis
            if st.session_state.text_data:
                try:
                    if model_choice == "VADER (local, fast)":
                        st.session_state.results_df = analyze_sentiments_vader(st.session_state.text_data)
                    elif model_choice == "Hugging Face (API, accurate)" and st.session_state.api_token:
                        st.session_state.results_df = analyze_sentiments_hf(
                            st.session_state.text_data, 
                            st.session_state.api_token
                        )
                    elif model_choice == "Compare Both" and st.session_state.api_token:
                        vader_df = analyze_sentiments_vader(st.session_state.text_data)
                        hf_df = analyze_sentiments_hf(
                            st.session_state.text_data, 
                            st.session_state.api_token
                        )
                        vader_df.columns = [col + " (VADER)" if col != "Text" else col for col in vader_df.columns]
                        hf_df.columns = [col + " (HF)" if col != "Text" else col for col in hf_df.columns]
                        st.session_state.results_df = pd.merge(
                            vader_df, 
                            hf_df[["Text", "Sentiment (HF)", "Confidence (HF)"]], 
                            on="Text"
                        )
                    else:
                        st.warning("API token required for Hugging Face or Compare Both mode.")
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
            else:
                st.session_state.results_df = pd.DataFrame()
                st.warning("No valid text data provided for analysis.")
            st.rerun()

# Text preprocessing function
def preprocess_text(texts, remove_stopwords=True, remove_punctuation=True, lowercase=True):
    stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    processed_texts = []
    for text in texts:
        if not isinstance(text, str):
            text = str(text)
        if lowercase:
            text = text.lower()
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        if remove_stopwords:
            tokens = word_tokenize(text)
            text = ' '.join([word for word in tokens if word not in stop_words])
        processed_texts.append(text)
    return processed_texts

# Main Content Area
if st.session_state.text_data and not st.session_state.results_df.empty:
    st.markdown('<div class="section-header">📊 Results</div>', unsafe_allow_html=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"**Timestamp**: {timestamp}")
    
    # Determine sentiment column based on model
    sentiment_column = "Sentiment"
    if st.session_state.model_choice == "Compare Both":
        sentiment_column = "Sentiment (VADER)"  # Use VADER as primary for metrics
    
    # Metrics
    if sentiment_column in st.session_state.results_df.columns:
        col1, col2, col3 = st.columns(3)
        with col1:
            positive_pct = (st.session_state.results_df[sentiment_column] == 'Positive').mean() * 100
            st.metric("Positive", f"{positive_pct:.1f}%")
        with col2:
            neutral_pct = (st.session_state.results_df[sentiment_column] == 'Neutral').mean() * 100
            st.metric("Neutral", f"{neutral_pct:.1f}%")
        with col3:
            negative_pct = (st.session_state.results_df[sentiment_column] == 'Negative').mean() * 100
            st.metric("Negative", f"{negative_pct:.1f}%")
    else:
        st.warning("Sentiment data not available for metrics.")

    # Export buttons
    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        st.download_button(
            "📄 CSV", 
            st.session_state.results_df.to_csv(index=False).encode(), 
            "sentiment_results.csv", 
            "text/csv"
        )
    with export_col2:
        st.download_button(
            "📝 JSON", 
            st.session_state.results_df.to_json(orient="records"), 
            "sentiment_results.json", 
            "application/json"
        )
    with export_col3:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            st.session_state.results_df.to_excel(writer, index=False)
        st.download_button(
            "📑 Excel", 
            buffer.getvalue(), 
            "sentiment_results.xlsx", 
            "application/vnd.ms-excel"
        )

    # Visualization Section
    tab1, tab2, tab3 = st.tabs(["📈 Distribution", "📉 Trends", "🌥️ Keyword Cloud"])
    
    with tab1:
        if sentiment_column in st.session_state.results_df.columns:
            fig = px.pie(
                st.session_state.results_df,
                names=sentiment_column,
                color=sentiment_column,
                color_discrete_map={
                    'Positive': '#50C878', 
                    'Neutral': '#7F8C8D', 
                    'Negative': '#FF6B6B'
                }
            )
            fig.update_layout(
                plot_bgcolor="#1C2526", 
                paper_bgcolor="#1C2526",
                font_color="#F5F6F5", 
                title_font_size=18
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid sentiment column for pie chart.")
    
    with tab2:
        if 'Confidence' in st.session_state.results_df.columns or \
           'Confidence (VADER)' in st.session_state.results_df.columns or \
           'Confidence (HF)' in st.session_state.results_df.columns:
            
            confidence_col = 'Confidence'
            if 'Confidence (VADER)' in st.session_state.results_df.columns:
                confidence_col = 'Confidence (VADER)'
            elif 'Confidence (HF)' in st.session_state.results_df.columns:
                confidence_col = 'Confidence (HF)'
            
            fig = px.line(
                st.session_state.results_df,
                x=st.session_state.results_df.index,
                y=confidence_col,
                color=sentiment_column,
                color_discrete_map={
                    'Positive': '#50C878', 
                    'Neutral': '#7F8C8D', 
                    'Negative': '#FF6B6B'
                }
            )
            fig.update_layout(
                plot_bgcolor="#1C2526", 
                paper_bgcolor="#1C2526",
                font_color="#F5F6F5", 
                title_font_size=18
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No confidence data available for trend visualization.")
    
    with tab3:
        if 'keywords' in st.session_state.results_df.attrs:
            keywords = st.session_state.results_df.attrs.get('keywords', [])
            if keywords:
                try:
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='#1C2526',
                        colormap='Blues', 
                        color_func=lambda *args, **kwargs: "#F5F6F5"
                    ).generate_from_frequencies(dict(keywords))
                    
                    fig, ax = plt.subplots(facecolor='#1C2526')
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating word cloud: {str(e)}")
            else:
                st.warning("No keywords available for word cloud.")
        else:
            st.warning("Keywords not extracted. Enable keyword extraction in settings.")

    # Confidence Gauges
    if sentiment_column in st.session_state.results_df.columns:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Positive Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number", 
                value=positive_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Positive %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#50C878'},
                    'steps': [{'range': [0, 100], 'color': '#2E3537'}]
                }
            ))
            fig.update_layout(paper_bgcolor="#1C2526", font_color="#F5F6F5")
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Neutral Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number", 
                value=neutral_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Neutral %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#7F8C8D'},
                    'steps': [{'range': [0, 100], 'color': '#2E3537'}]
                }
            ))
            fig.update_layout(paper_bgcolor="#1C2526", font_color="#F5F6F5")
            st.plotly_chart(fig)
        
        with col3:
            st.subheader("Negative Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number", 
                value=negative_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Negative %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#FF6B6B'},
                    'steps': [{'range': [0, 100], 'color': '#2E3537'}]
                }
            ))
            fig.update_layout(paper_bgcolor="#1C2526", font_color="#F5F6F5")
            st.plotly_chart(fig)

    # Detailed Analysis Panel (collapsible)
    with st.expander("📋 Detailed Analysis", expanded=False):
        if not st.session_state.results_df.empty:
            for idx, row in st.session_state.results_df.iterrows():
                st.markdown(f"**Text {idx + 1}:** {row['Text']}")
                
                sentiment_val = row.get(sentiment_column, "N/A")
                confidence_val = row.get('Confidence', row.get('Confidence (VADER)', row.get('Confidence (HF)', 0.0)))
                
                st.markdown(f"**Sentiment:** {sentiment_val} (Confidence: {confidence_val:.2f})")
                
                if 'keywords' in st.session_state.results_df.attrs:
                    keywords = st.session_state.results_df.attrs.get('keywords', [])
                    if keywords:
                        highlighted = ' '.join([
                            f"<mark>{word}</mark>" if any(kw[0].lower() in word.lower() for kw in keywords) 
                            else word 
                            for word in row['Text'].split()
                        ])
                        st.markdown(f"**Highlighted Keywords:** {highlighted}", unsafe_allow_html=True)
                
                st.markdown(
                    f"**Explanation:** Sentiment determined by {sentiment_val.lower() if sentiment_val != 'N/A' else 'unknown'} language cues.", 
                    unsafe_allow_html=True
                )
                st.markdown("---")

    # Comparative Analysis Section (collapsible)
    with st.expander("🔍 Comparative Analysis", expanded=False):
        selected_texts = st.multiselect(
            "Select Texts to Compare", 
            st.session_state.text_data, 
            key="compare_select"
        )
        st.session_state.selected_texts = selected_texts
        
        if len(selected_texts) >= 2:
            try:
                comp_df = pd.DataFrame()
                if model_choice == "VADER (local, fast)":
                    comp_df = analyze_sentiments_vader(selected_texts)
                elif model_choice == "Hugging Face (API, accurate)" and st.session_state.api_token:
                    comp_df = analyze_sentiments_hf(selected_texts, st.session_state.api_token)
                elif model_choice == "Compare Both" and st.session_state.api_token:
                    vader_df = analyze_sentiments_vader(selected_texts)
                    hf_df = analyze_sentiments_hf(selected_texts, st.session_state.api_token)
                    vader_df.columns = [col + " (VADER)" if col != "Text" else col for col in vader_df.columns]
                    hf_df.columns = [col + " (HF)" if col != "Text" else col for col in hf_df.columns]
                    comp_df = pd.merge(
                        vader_df, 
                        hf_df[["Text", "Sentiment (HF)", "Confidence (HF)"]], 
                        on="Text"
                    )
                
                if not comp_df.empty:
                    fig = go.Figure()
                    for i, text in enumerate(selected_texts):
                        row = comp_df[comp_df['Text'] == text].iloc[0]
                        sentiment = row.get(sentiment_column, "Neutral")
                        confidence = row.get('Confidence', row.get('Confidence (VADER)', row.get('Confidence (HF)', 0.0)))
                        
                        sentiment_score = confidence * (
                            1 if sentiment == 'Positive' else 
                            -1 if sentiment == 'Negative' else 
                            0
                        )
                        
                        fig.add_trace(go.Bar(
                            x=[f"Text {i+1}"],
                            y=[sentiment_score],
                            name=text[:50] + "..." if len(text) > 50 else text,
                            text=[text],
                            hoverinfo="text+name"
                        ))
                    
                    fig.update_layout(
                        plot_bgcolor="#1C2526", 
                        paper_bgcolor="#1C2526",
                        font_color="#F5F6F5", 
                        barmode='group',
                        title="Sentiment Comparison"
                    )
                    st.plotly_chart(fig)
                    
                    if len(comp_df) >= 2:
                        conf1 = comp_df.iloc[0].get('Confidence', comp_df.iloc[0].get('Confidence (VADER)', comp_df.iloc[0].get('Confidence (HF)', 0.0)))
                        conf2 = comp_df.iloc[1].get('Confidence', comp_df.iloc[1].get('Confidence (VADER)', comp_df.iloc[1].get('Confidence (HF)', 0.0)))
                        diff = abs(conf1 - conf2)
                        st.metric("Confidence Difference", f"{diff:.2f}")
            except Exception as e:
                st.error(f"Comparative analysis error: {str(e)}")

    # Error Handling and Status
    st.markdown('<div class="section-header">⚠️ Status</div>', unsafe_allow_html=True)
    if st.session_state.text_data:
        if st.session_state.results_df.empty:
            st.warning("No analysis results available. Please ensure data is processed.")
        else:
            st.success(f"Analysis completed at {datetime.now().strftime('%H:%M:%S')} with {len(st.session_state.results_df)} texts.")
    else:
        st.warning("No text data provided for analysis.")
    
    if model_choice in ["Hugging Face (API, accurate)", "Compare Both"] and not st.session_state.api_token:
        st.error("API token required for Hugging Face or Compare Both mode.")

# Documentation Access
with st.sidebar:
    if st.button("ℹ️ Help", key="help_button"):
        st.sidebar.markdown("""
        ### Quick Guide
        - **Input**: Enter text or upload a file.
        - **Analyze**: Click 'Analyze' to process data.
        - **Settings**: Adjust model, threshold, and keyword options.
        - **Visuals**: Explore pie charts, trends, and keyword clouds.
        - **Compare**: Select texts for comparative analysis.
        
        **Model Limitations**: 
        - VADER is fast but less accurate
        - Hugging Face requires an API token and internet connection
        
        **Confidence Threshold**: Filters results below the set value.
        """, unsafe_allow_html=True)

    # Data Persistence
    if st.button("💾 Save Analysis"):
        if not st.session_state.results_df.empty:
            try:
                with open("sentiment_analysis.json", "w") as f:
                    json.dump({
                        'results': st.session_state.results_df.to_dict(),
                        'timestamp': datetime.now().isoformat()
                    }, f)
                st.success("Analysis saved to sentiment_analysis.json")
            except Exception as e:
                st.error(f"Error saving analysis: {str(e)}")
        else:
            st.warning("No analysis results to save.")
    
    if st.button("📂 Load Analysis"):
        try:
            if os.path.exists("sentiment_analysis.json"):
                with open("sentiment_analysis.json", "r") as f:
                    data = json.load(f)
                    st.session_state.results_df = pd.DataFrame(data['results'])
                st.success(f"Analysis loaded from sentiment_analysis.json (saved at {data.get('timestamp', 'unknown time')})")
            else:
                st.warning("No saved analysis found.")
        except Exception as e:
            st.error(f"Error loading analysis: {str(e)}")

# Real-time Updates
if st.session_state.text_data and st.button("🔄 Refresh", key="refresh_button"):
    with st.spinner("Refreshing..."):
        time.sleep(1)  # Simulate processing
        try:
            if model_choice == "VADER (local, fast)":
                st.session_state.results_df = analyze_sentiments_vader(st.session_state.text_data)
            elif model_choice == "Hugging Face (API, accurate)" and st.session_state.api_token:
                st.session_state.results_df = analyze_sentiments_hf(st.session_state.text_data, st.session_state.api_token)
            elif model_choice == "Compare Both" and st.session_state.api_token:
                vader_df = analyze_sentiments_vader(st.session_state.text_data)
                hf_df = analyze_sentiments_hf(st.session_state.text_data, st.session_state.api_token)
                vader_df.columns = [col + " (VADER)" if col != "Text" else col for col in vader_df.columns]
                hf_df.columns = [col + " (HF)" if col != "Text" else col for col in hf_df.columns]
                st.session_state.results_df = pd.merge(
                    vader_df, 
                    hf_df[["Text", "Sentiment (HF)", "Confidence (HF)"]], 
                    on="Text"
                )
            st.rerun()
        except Exception as e:
            st.error(f"Refresh error: {str(e)}")

elif not st.session_state.text_data:
    st.markdown(
        '<div class="info-box">Please enter text or upload a file to start analyzing.</div>', 
        unsafe_allow_html=True
    )