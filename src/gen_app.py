import streamlit as st
import PyPDF2  # Or Tika if preferred
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import plotly.express as px # Or Altair if preferred


# I. Data Input and Handling

def upload_pdf():
    """Handles PDF file upload and validation."""
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is None:
        st.error("Please upload a PDF file.")
        return None
    try:
        return uploaded_file.read()
    except Exception as e:
        st.error(f"Error uploading file: {e}")
        return None

def extract_text(pdf_bytes):
    """Extracts text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(pdf_bytes)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except PyPDF2.errors.PdfReadError as e:
        st.error(f"Error reading PDF file: {e}. It may be corrupted or password-protected.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during PDF processing: {e}")
        return None


def validate_text(text):
    """Validates extracted text."""
    if not text.strip():
        st.error("The PDF file appears to be empty or contains only non-textual data.")
        return None
    return text


# II. TF-IDF Calculation and Analysis

def preprocess_text(text):
    """Preprocesses the extracted text."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        tokens = nltk.word_tokenize(text)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tokens = [token.lower() for token in tokens if token.isalnum() and token.lower() not in stop_words]
        return " ".join(tokens)
    except Exception as e:
        st.error(f"An error occurred during text preprocessing: {e}")
        return None


def calculate_tfidf(text):
    """Calculates TF-IDF scores."""
    try:
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=(1, 2)) # Adjust parameters as needed
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        return feature_names, tfidf_scores
    except Exception as e:
        st.error(f"An error occurred during TF-IDF calculation: {e}")
        return None, None


def rank_frequencies(feature_names, tfidf_scores):
    """Ranks terms by TF-IDF scores."""
    if feature_names is None or tfidf_scores is None:
        return pd.DataFrame() #Return empty dataframe if error in calculation
    df = pd.DataFrame({'term': feature_names, 'tfidf': tfidf_scores})
    df = df.sort_values('tfidf', ascending=False)
    return df


# III. Output and Presentation

def display_results(df):
    """Displays the results using Streamlit."""
    if df.empty:
        st.info("No results to display. Please check the uploaded PDF file.")
        return

    st.subheader("Top Words by TF-IDF Score")
    st.dataframe(df.head(20)) # Show top 20
    fig = px.bar(df.head(20), x='term', y='tfidf', title="Top 20 Words")
    st.plotly_chart(fig)

# Main application logic
st.title("PDF TF-IDF Analyzer")

pdf_bytes = upload_pdf()
if pdf_bytes:
    extracted_text = extract_text(pdf_bytes)
    if extracted_text:
        cleaned_text = validate_text(extracted_text)
        if cleaned_text:
            preprocessed_text = preprocess_text(cleaned_text)
            if preprocessed_text:
                feature_names, tfidf_scores = calculate_tfidf(preprocessed_text)
                df = rank_frequencies(feature_names, tfidf_scores)
                display_results(df)