"""
AI Biography Chatbot
--------------------
A semantic chatbot designed to answer questions about famous biographies 
using natural language processing (NLP) techniques.

Text Sources: Curated biographies derived from Project Gutenberg archives.

Framework: Streamlit | NLP: NLTK & Scikit-Learn

Author: Tonia M. Ethuakhor

Date: February 2026
"""


# 1. Import Libraries
# -------------------
# Import necessary libraries for the web app, text processing, and similarity analysis
import streamlit as st
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 2. Environment Setup And NLP Initialisation
# -------------------------------------------
# This section sets up the Streamlit app and downloads the language tools needed to work with text.

st.set_page_config(page_title="AI Biography Chatbot", page_icon="🤖")

@st.cache_resource
def download_nltk_resources():
    """Downloads NLTK tools used for splitting text and removing common words."""
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

download_nltk_resources()
stemmer = PorterStemmer()  # Reduces words to their root form (e.g. 'reading' ==> 'read')

st.title("🤖 AI Biography Assistant")
st.markdown("Explore the lives of **Helen Keller** and **Nikola Tesla** using semantic search.")


# 3. Data Preprocessing And Text Cleaning
# ---------------------------------------
def preprocess(text):
    """
    Cleans and prepares text so it is easier for the computer to compare.
    The text is lowercased, common words are removed, and words are reduced
    to their base form. Hyphens are kept so terms like 'co-found' still match.
    """
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    
    filtered_words = []
    for w in words:
        if (w.isalnum() or '-' in w) and w not in stop_words:
            filtered_words.append(stemmer.stem(w))
            
    return " ".join(filtered_words)


# 4. Knowledge Indexing And Tf-Idf Vectorisation
# ----------------------------------------------
@st.cache_data
def get_indexed_knowledge(filenames):
    """
    Reads the text files, splits them into sentences, and converts each
    sentence into numbers using TF-IDF so the chatbot can compare meaning.
    """
    all_sentences = []
    
    for fname in filenames:
        if os.path.exists(fname):
            with open(fname, 'r', encoding='utf-8') as f:
                content = f.read()
            sentences = [s.strip() for s in sent_tokenize(content) if len(s) > 10]
            all_sentences.extend(sentences)
    
    if not all_sentences:
        return None, None, None
        
    processed_sentences = [preprocess(s) for s in all_sentences]
    
    # Turn each sentence into numbers so the computer can compare their meaning
    # Using 1- to 3-word combinations helps it recognise names and important phrases
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)
    
    return all_sentences, vectorizer, tfidf_matrix


# 5. Semantic Search And Similarity Matching
# ------------------------------------------
def chatbot_response(query, sentences, vectorizer, tfidf_matrix):
    """
    Takes the user's question and checks how similar it is to each sentence
    in the knowledge base using cosine similarity.
    """
    processed_query = preprocess(query)
    
    if not processed_query:
        return "Please ask a specific question!", 0.0

    # Convert the user's question into numerical form
    query_vec = vectorizer.transform([processed_query])
    
    # Compare the question with all stored sentences
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    
    # Find the sentence with the highest similarity score
    idx = scores.argmax()
    confidence = scores[idx]
    
    if confidence < 0.10:
        return (
            "I'm sorry, I could not find a clear match in the text. "
            "Try using different keywords!",
            confidence
        )
    
    return sentences[idx], confidence


# 6. Streamlit Interface And User Interaction
# -------------------------------------------
def main():
    # Sidebar allows the user to choose which biography to search
    st.sidebar.header("🗄️ Library Settings")
    option = st.sidebar.selectbox(
        "Which biography would you like to search?",
        ("Helen Keller Only", "Nikola Tesla Only", "Both Biographies")
    )
    
    # Match the user's choice to the correct text files
    file1, file2 = "helen_keller.txt", "nikola_tesla.txt"
    if option == "Helen Keller Only":
        files_to_load = [file1]
    elif option == "Nikola Tesla Only":
        files_to_load = [file2]
    else:
        files_to_load = [file1, file2]

    # Load and process the text files (this runs only once due to caching)
    sentences, vectorizer, tfidf_matrix = get_indexed_knowledge(tuple(files_to_load))

    # Safety check: if the text could not be processed, stop the app here
    # instead of allowing errors to happen later.
    if sentences is None or vectorizer is None:
        st.error(
            "The knowledge base could not be loaded. "
            "Please check that the text files exist and contain valid content."
        )
        return

    # Suggested questions help users get started quickly
    st.sidebar.markdown("---")
    st.sidebar.subheader("Suggested Questions")
    
    suggested_q = ""
    
    # Questions related to Helen Keller
    if "Helen" in option or option == "Both Biographies":
        st.sidebar.write("**Explore Helen Keller:**")
        hk_qs = [
            "Where was Helen Keller born?",
            "How did she learn the word for water?",
            "Who was Anne Sullivan?",
            "How did she learn to speak?",
            "What was her college degree?",
            "Who was Alexander Graham Bell?",
            "What organization did she co-found?",
            "How many countries did she visit?",
            "What is the title of her autobiography?",
            "When did Helen Keller die?"
        ]
        for q in hk_qs:
            if st.sidebar.button(q, key=f"hk_{q}"):
                suggested_q = q
        
    # Questions related to Nikola Tesla
    if "Tesla" in option or option == "Both Biographies":
        st.sidebar.write("**Explore Nikola Tesla:**")
        nt_qs = [
            "When was Nikola Tesla born?",
            "What is he best known for?",
            "Who was Thomas Edison to him?",
            "What was the War of Currents?",
            "What was his first major invention?",
            "What is the Tesla Coil?",
            "Who was George Westinghouse?",
            "What happened at Niagara Falls?",
            "How did he visualise his inventions?",
            "How many languages did he speak?"
        ]
        for q in nt_qs:
            if st.sidebar.button(q, key=f"nt_{q}"):
                suggested_q = q

    # Main area where the chatbot answers questions
    st.sidebar.success(f"{len(sentences)} sentences indexed.")
    
    user_query = st.text_input("Ask a question about the biography:", value=suggested_q)
    
    if user_query:
        with st.spinner("Searching the knowledge base..."):
            answer, confidence = chatbot_response(
                user_query, sentences, vectorizer, tfidf_matrix
            )
            
            st.subheader("Chatbot Answer:")
            st.success(answer)
            st.info(f"Similarity Confidence: {confidence:.2f}")
            st.caption(
                "Note: A higher confidence score means the answer is a closer match "
                "to what you asked."
            )

    # How the Bot Thinks and Finds Answers
    st.sidebar.markdown("---")
    with st.sidebar.expander("How does this work?"):
        st.write("""
            **How Your Questions Are Matched:**
            - **TF-IDF:** Converts sentences into numbers based on word importance.
            - **Cosine Similarity:** Measures the 'angle' between your question and the text to find the closest match.
            - **Stemming:** Chops words to their roots (e.g., 'founder' ==> 'found') to improve matching accuracy.
        """)

if __name__ == "__main__":
    main()