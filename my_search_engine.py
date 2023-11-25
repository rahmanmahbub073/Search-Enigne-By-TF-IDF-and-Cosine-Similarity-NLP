# my_search_engine.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string


class MySearchEngine:
    def __init__(self):
        # Load the DBLP dataset (Assuming it's in CSV format with columns 'author', 'title', 'year')
        self.dblp_dataset_path = 'datasets/Final_paperforsearch2.csv'
        self.df_dblp = pd.read_csv(self.dblp_dataset_path, usecols=['author', 'ptitle', 'year'], nrows=99997)

        # Combine relevant columns into a single text column for processing
        self.df_dblp['text'] = self.df_dblp['author'] + ' ' + self.df_dblp['ptitle'] + ' ' + self.df_dblp['year'].astype(str)

        # Preprocess the dataset
        self.df_dblp['processed_text'] = self.df_dblp['text'].apply(self.preprocess_text)

        # Train the TF-IDF algorithm
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df_dblp['processed_text'])

    def preprocess_text(self, text):
        # Tokenize the text
        words = word_tokenize(text.lower())

        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.isalnum() and word not in stop_words and word not in string.punctuation]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        return ' '.join(words)

    def extract_suggestions(self, sentence, num_documents=20, similarity_threshold=0.3):
        print("Starting suggestion extraction")
        # Preprocess the input sentence
        processed_sentence = self.preprocess_text(sentence)

        # Transform the input sentence using the trained TF-IDF model
        sentence_vector = self.tfidf_vectorizer.transform([processed_sentence])

        # Compute cosine similarity between the input sentence and each document in the dataset
        similarities = cosine_similarity(sentence_vector, self.tfidf_matrix)

        # Get the indices of documents sorted by similarity in descending order
        similar_indices = similarities.argsort()[0][::-1]

        # Extract suggestions from the most similar documents
        result = []
        for idx in similar_indices[:num_documents]:
            title = self.df_dblp.loc[idx, 'ptitle']
            author = self.df_dblp.loc[idx, 'author']
            year = self.df_dblp.loc[idx, 'year']
            similarity_score = similarities[0, idx]

            # Only include suggestions with similarity scores above the threshold
            if similarity_score > similarity_threshold:
                # Extract keywords for the current document
                keywords = self.extract_keywords_for_document(idx)
                result.append({'ptitle': title, 'author': author, 'year': year, 'similarity': similarity_score, 'keywords': keywords})

        print("Finished suggestion extraction")

        return result

    def extract_keywords_for_document(self, document_idx, num_keywords=5):
        # Get TF-IDF representation for the specific document
        document_text = self.df_dblp.loc[document_idx, 'processed_text']
        document_vector = self.tfidf_vectorizer.transform([document_text])

        # Get feature names (words) from the TF-IDF model
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        # Get TF-IDF scores for each feature in the document
        tfidf_scores = document_vector.toarray()[0]

        # Combine feature names with their TF-IDF scores
        feature_tfidf_scores = list(zip(feature_names, tfidf_scores))

        # Sort features by TF-IDF scores in descending order
        sorted_features = sorted(feature_tfidf_scores, key=lambda x: x[1], reverse=True)

        # Extract top keywords for the document
        top_keywords = [feature[0] for feature in sorted_features[:num_keywords]]

        return top_keywords


    def extract_keywords(self):
            # Combine 'ptitle', 'author', and 'year' into a single text column for keyword extraction
            self.df_dblp['keywords_text'] = self.df_dblp['ptitle'] + ' ' + self.df_dblp['author'] + ' ' + self.df_dblp['year'].astype(str)

            # Preprocess the text for keyword extraction
            self.df_dblp['processed_keywords'] = self.df_dblp['keywords_text'].apply(self.preprocess_text)

            # Train a new TF-IDF model for keyword extraction
            tfidf_vectorizer_keywords = TfidfVectorizer()
            tfidf_matrix_keywords = tfidf_vectorizer_keywords.fit_transform(self.df_dblp['processed_keywords'])

            # Get feature names (words) from the TF-IDF model
            feature_names = tfidf_vectorizer_keywords.get_feature_names_out()

            # Extract top keywords for each document
            all_keywords = []
            for idx, row in enumerate(tfidf_matrix_keywords):
                # Get indices of top keywords based on TF-IDF scores
                top_keyword_indices = row.indices

                # Map indices to actual words
                top_keyword_words = [feature_names[idx] for idx in top_keyword_indices]

                all_keywords.append({'ptitle': self.df_dblp.loc[idx, 'ptitle'], 'author': self.df_dblp.loc[idx, 'author'],
                                    'year': self.df_dblp.loc[idx, 'year'], 'keywords': top_keyword_words})

            # Return a list of dictionaries with document information and top keywords
            return all_keywords