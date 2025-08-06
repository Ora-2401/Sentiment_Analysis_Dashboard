import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from huggingface_hub import InferenceClient
from sklearn.feature_extraction.text import CountVectorizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log package versions for debugging (only for huggingface_hub, as vaderSentiment lacks __version__)
try:
    import huggingface_hub
    logger.info(f"huggingface_hub version: {huggingface_hub.__version__}")
except ImportError as e:
    logger.error(f"Import error: {str(e)}")

def classify_sentiment_vader(compound):
    """Classify VADER compound score into sentiment categories."""
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def extract_keywords(texts, top_n=10):
    """Extract top N keywords from texts using CountVectorizer."""
    try:
        if not texts or all(not text.strip() for text in texts):
            logger.warning("No valid texts provided for keyword extraction.")
            return []
        vec = CountVectorizer(stop_words='english')
        X = vec.fit_transform(texts)
        sum_words = X.sum(axis=0)
        keywords = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        return sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return []

def analyze_sentiments_vader(texts):
    """Analyze sentiments using VADER."""
    if not texts:
        logger.warning("Empty text list provided to VADER analysis.")
        return pd.DataFrame(columns=["Text", "Sentiment", "Positive Score", "Neutral Score", "Negative Score", "Confidence", "Compound Score"])

    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for text in texts:
        if not text.strip():
            rows.append({
                "Text": text,
                "Sentiment": "Neutral",
                "Positive Score": 0.0,
                "Neutral Score": 1.0,
                "Negative Score": 0.0,
                "Confidence": 1.0,
                "Compound Score": 0.0
            })
            continue
        try:
            scores = analyzer.polarity_scores(text)
            sentiment = classify_sentiment_vader(scores['compound'])
            rows.append({
                "Text": text,
                "Sentiment": sentiment,
                "Positive Score": scores['pos'],
                "Neutral Score": scores['neu'],
                "Negative Score": scores['neg'],
                "Confidence": max(scores['pos'], scores['neu'], scores['neg']),
                "Compound Score": scores['compound']
            })
        except Exception as e:
            logger.error(f"Error processing VADER text '{text}': {str(e)}")
            rows.append({
                "Text": text,
                "Sentiment": "Error",
                "Positive Score": 0.0,
                "Neutral Score": 0.0,
                "Negative Score": 0.0,
                "Confidence": 0.0,
                "Compound Score": 0.0
            })

    df = pd.DataFrame(rows)
    df.attrs['keywords'] = extract_keywords([text for text in texts if text.strip()])
    return df

def analyze_sentiments_hf(texts, api_token):
    """Analyze sentiments using Hugging Face Inference API."""
    if not texts:
        logger.warning("Empty text list provided to HF analysis.")
        return pd.DataFrame(columns=["Text", "Sentiment", "Confidence"])
    if not api_token:
        logger.error("No API token provided for Hugging Face analysis.")
        raise ValueError("Hugging Face API token is required.")

    client = InferenceClient(token=api_token)
    # Updated model to a known working Inference API model
    model = "distilbert-base-uncased-finetuned-sst-2-english"  # Original model
    # Fallback to a verified model if the original fails
    fallback_model = "nlptown/bert-base-multilingual-uncased-sentiment"
    
    results = []
    for text in texts:
        if not text.strip():
            results.append({
                "Text": text,
                "Sentiment": "Neutral",
                "Confidence": 1.0
            })
            continue
        try:
            # Attempt with original model
            result = client.text_classification(text, model=model)
            label = result[0]['label']  # Adjust for API response format
            score = result[0]['score']
            sentiment = 'Positive' if label == 'POSITIVE' else 'Negative' if label == 'NEGATIVE' else 'Neutral'
            results.append({
                "Text": text,
                "Sentiment": sentiment,
                "Confidence": round(score, 3)
            })
        except Exception as e:
            logger.error(f"Error processing HF text '{text}' with {model}: {str(e)}")
            try:
                # Fallback to alternative model
                result = client.text_classification(text, model=fallback_model)
                label = result[0]['label']
                score = result[0]['score']
                sentiment = 'Positive' if label == 'POSITIVE' else 'Negative' if label == 'NEGATIVE' else 'Neutral'
                results.append({
                    "Text": text,
                    "Sentiment": sentiment,
                    "Confidence": round(score, 3)
                })
            except Exception as e2:
                logger.error(f"Error processing HF text '{text}' with {fallback_model}: {str(e2)}")
                results.append({
                    "Text": text,
                    "Sentiment": "Error",
                    "Confidence": 0.0
                })

    df = pd.DataFrame(results)
    df.attrs['keywords'] = extract_keywords([text for text in texts if text.strip()])
    return df

if __name__ == "__main__":
    # Example usage for testing
    sample_texts = ["I love this product!", "This is terrible.", ""]
    vader_df = analyze_sentiments_vader(sample_texts)
    print("VADER Results:\n", vader_df)
    # Note: Requires a valid API token for HF
    # hf_df = analyze_sentiments_hf(sample_texts, "your_api_token_here")
    # print("Hugging Face Results:\n", hf_df)