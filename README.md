# Sentiment_Analysis_Dashboard
## Overview
The **Sentiment Explorer Dashboard** is a Streamlit-based web application designed for analyzing the emotional tone of text data. It supports sentiment analysis using VADER (local, fast) and Hugging Face (API-based, accurate) models, with options for comparing both. The dashboard provides interactive visualizations, keyword extraction, and data export functionalities, all wrapped in a modern, dark-themed UI.

## Access the Application
You can access the deployed application [here](https://sentimentanalysisdashboard-ds09.onrender.com/).

## Video Demonstration
Watch a video demonstration of the application [here](https://capeitinitiative-my.sharepoint.com/:v:/g/personal/lesego_mphahlele_capaciti_org_za/ESrXTxzCrRJCtLdlK9gvF9EBbI20r2HPK6bTaCiWqe--BQ?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=LWTWbx).

## Features
- **Text Input**: Supports manual text entry (one item per line) or file uploads (CSV/TXT).
- **Sentiment Analysis**:
  - **VADER**: Fast, local sentiment analysis.
  - **Hugging Face**: Accurate, API-based analysis (requires API token).
  - **Compare Both**: Side-by-side comparison of VADER and Hugging Face results.
- **Visualizations**:
  - Pie chart for sentiment distribution.
  - Line chart for confidence trends.
  - Word cloud for keyword visualization.
  - Gauge charts for sentiment percentages (Positive, Neutral, Negative).
- **Comparative Analysis**: Compare sentiments of selected texts with bar charts.
- **Data Export**: Save results as CSV, JSON, or Excel files.
- **Data Persistence**: Save and load analysis results to/from a JSON file.
- **Customizable Settings**:
  - Confidence threshold for filtering results.
  - Option to include/exclude neutral sentiments.
  - Keyword extraction toggle.
- **Interactive UI**: Dark theme with animations, tooltips, and responsive design.

## Requirements
To run the application, ensure you have the following dependencies installed:

```bash
streamlit
pandas
plotly
wordcloud
matplotlib
nltk
xlsxwriter
```

Install them using:

```bash
pip install streamlit pandas plotly wordcloud matplotlib nltk xlsxwriter
```

Additionally, for Hugging Face model usage, you need a valid Hugging Face API token.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Run the pip install command above to set up the required Python packages.

3. **Download NLTK Data**:
   The application automatically downloads required NLTK data (`punkt` and `stopwords`) on first run. Alternatively, you can manually download them:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

4. **Run the Application**:
   Launch the Streamlit app with:
   ```bash
   streamlit run app.py
   ```
   Replace `app.py` with the name of your Python script containing the provided code.

5. **Optional: Hugging Face API Token**:
   If using the Hugging Face model or comparison mode, obtain an API token from [Hugging Face](https://huggingface.co/) and enter it in the sidebar when prompted.

## Usage
1. **Input Data**:
   - Select the data source type (Reviews, Social Media, Custom).
   - Enter text manually in the text area (one item per line) or upload a CSV/TXT file.
   - For CSV files, choose the column containing text data.

2. **Configure Analysis**:
   - Choose a model: VADER, Hugging Face, or Compare Both.
   - Set the confidence threshold (0.0 to 1.0).
   - Enable/disable keyword extraction and neutral sentiment inclusion.

3. **Analyze**:
   - Click the "Analyze" button to process the text.
   - View results in the main panel, including sentiment metrics, visualizations, and detailed analysis.

4. **Visualizations**:
   - **Distribution Tab**: Pie chart showing sentiment distribution.
   - **Trends Tab**: Line chart of confidence scores over texts.
   - **Keyword Cloud Tab**: Word cloud of extracted keywords (if enabled).

5. **Export and Save**:
   - Download results as CSV, JSON, or Excel.
   - Save the analysis to a JSON file for later retrieval.
   - Load previous analyses from saved JSON files.

6. **Comparative Analysis**:
   - Select multiple texts to compare their sentiments in a bar chart.
   - View confidence differences between selected texts.

7. **Refresh**: Re-run the analysis with current settings using the "Refresh" button.

## File Structure
- `app.py`: Main application script (rename as needed).
- `components/`: Directory containing helper modules (`sentiment_analysis.py`, `utils.py`, `visuals.py`).
- `sentiment_analysis.json`: Output file for saved analyses (generated after saving).

## Notes
- **Model Limitations**:
  - VADER is fast but less accurate, suitable for quick analyses.
  - Hugging Face requires an internet connection and a valid API token for higher accuracy.
- **Error Handling**: The app includes robust error handling for file uploads, API issues, and analysis Failures.
- **Performance**: Processing time depends on the input size and model choice. Hugging Face may be slower due to API calls.
- **Keyword Extraction**: Requires NLTK's `punkt` and `stopwords` for preprocessing.

## Example
To analyze customer reviews:
1. Select "Reviews" as the data source.
2. Upload a CSV file with a column of review texts or paste reviews in the text area.
3. Choose "Compare Both" for VADER and Hugging Face analysis.
4. Enter your Hugging Face API token.
5. Adjust settings (e.g., confidence threshold = 0.6, enable keyword extraction).
6. Click "Analyze" to view results, including visualizations and downloadable reports.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with [Streamlit](https://streamlit.io/) for the interactive UI.
- Uses [尾ADER](https://github.com/cjhutto/vaderSentiment) for local sentiment analysis.
- Integrates [Hugging Face](https://huggingface.co/) for advanced sentiment analysis.
- Visualizations powered by [Plotly](https://plotly.com/) and [WordCloud](https://github.com/amueller/word_cloud).
