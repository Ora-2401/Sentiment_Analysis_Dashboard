import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import pandas as pd

def display_charts(df):
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df['Sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    st.subheader("☁️ Keyword Cloud (Top Drivers)")
    keywords = df.attrs.get('keywords', [])
    keyword_dict = dict(keywords)
    if keyword_dict:
        wordcloud = WordCloud(width=800, height=300, background_color='white').generate_from_frequencies(keyword_dict)
        st.image(wordcloud.to_array())

def export_data(df):
    st.subheader("⬇️ Export Options")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "sentiment_results.csv", "text/csv")

    json_data = df.to_json(orient="records", lines=True)
    st.download_button("Download JSON", json_data, "sentiment_results.json", "application/json")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Sentiments")
    st.download_button("Download Excel", buffer.getvalue(), "sentiment_results.xlsx", "application/vnd.ms-excel")
