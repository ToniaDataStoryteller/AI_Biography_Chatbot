# 🤖 AI Biography Assistant: Historical Search Engine

An interactive Natural Language Processing (NLP) web application that retrieves facts from the lives of two iconic historical figures:

* Helen Keller
* Nikola Tesla

This project demonstrates the full engineering lifecycle - from research experimentation to production-ready deployment - with transparent documentation of the NLP methodologies involved.

## Knowledge Base

The AI Assistant’s data is curated from:

* *The Story of My Life* by Helen Keller
* *My Inventions* by Nikola Tesla

These texts are processed into searchable formats to allow semantic question answering.

## Features

### Dual Interaction Modes

#### 1️ Guided Exploration (Sidebar Interface)

* Select a specific biography or search both
* 20 interactive question buttons (10 per figure)
* Instant retrieval of key historical events

#### 2️ Natural Language Search

* Ask custom questions freely
* Semantic similarity matching (not just keyword search)
* Returns the most relevant sentence
* Displays a similarity score for transparency

## Technical Architecture

This repository reflects two development phases:

### Research Phase (`creating_a_chatbot.ipynb`)

Initial experimentation conducted in Jupyter Notebook:

* Explored transformer-based embeddings using `Sentence-Transformers`
* Tested model: `all-MiniLM-L6-v2`
* Experimented with semantic vector representations
* Validated similarity calculations

This phase prioritised exploration and validation of NLP strategies.

### Production Application (`chatbot_app.py`)

The deployed Streamlit web app includes:

#### Text Preprocessing

* Tokenization
* Porter Stemming
* Stop-word handling

#### Vectorization Strategy

* TF-IDF Vectorizer
* ngram_range = (1,3) to capture phrases and names
* Cosine Similarity for relevance scoring

#### Optimization

* Streamlit caching for sub-second responses
* Efficient sentence chunking for fast retrieval

## How the Search Works

1. The biography text is split into sentences.
2. Each sentence is converted into a numerical representation.
3. Users questions are also converted into numbers.
4. The system calculates similarity between the user's question and every sentence.
5. The most similar sentence is returned as the answer.

This allows the AI assistant to match meaning - not just exact words.

## Project Structure

``` text
AI_Biography_Assistant/
│
├── chatbot_app.py
├── creating_a_chatbot.ipynb
├── helen_keller.txt
├── nikola_tesla.txt
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:

git clone https://github.com/ToniaDataStoryteller/AI_Biography_Chatbot.git
cd AI_Biography_Chatbot



2. Install dependencies:

pip install -r requirements.txt



3. Run the application:

streamlit run chatbot_app.py


After launching the application, Streamlit will provide a local development URL (typically http://localhost:8501). Open this address in a web browser to access the interface.

## Example Questions

* When was Helen Keller born?
* What challenges did Helen Keller overcome?
* What inventions is Nikola Tesla known for?
* How did Tesla contribute to electrical engineering?

## Future Improvements

* Add additional historical figures
* Upgrade to transformer-based embeddings in production
* Deploy publicly via Streamlit Cloud
* Add conversational memory for follow-up questions
* Improve UI styling and mobile responsiveness


## Glossary

**Natural Language Processing (NLP):**
A field of Artificial Intelligence (AI), focused on enabling computers to understand human language.

**Tokenization:**
Breaking text into smaller pieces (usually words or sentences).

**Stemming:**
Reducing words to their root form (e.g., “invented” ==> “invent”).

**TF-IDF (Term Frequency–Inverse Document Frequency):**
A method that measures how important a word is in a piece of text compared to other text. 
 (e.g., Common words like "the" get a low score, while unique names like "Tesla" get a high score to help the AI find specific facts.)*

**n-grams:**
Sequences of words grouped together (e.g., “Nikola Tesla” is a 2-gram).

**Vectorization:**
Converting text into numerical form so a computer can perform mathematical comparisons.

**Cosine Similarity:**
A mathematical method for measuring how similar two pieces of text are.

**Semantic Search:**
Searching by meaning rather than exact word matches.

**Embedding:**
A high-dimensional numerical representation of text used in machine learning models.

## Learning Outcomes

This project demonstrates:

* Applied NLP techniques
* Text preprocessing pipelines
* Similarity-based search systems
* Web deployment using Streamlit
* Git-based version control
* Transition from research experimentation to production optimization

## License

This project is intended for educational and portfolio purposes.
