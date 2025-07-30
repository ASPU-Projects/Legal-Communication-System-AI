import nltk 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tashaphyne.stemming import ArabicLightStemmer
import re
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@staticmethod
def split_by_language(text):
  arabic= " ".join(re.findall(r"[\u0600-\u06FF]+", text))
  english = " ".join(re.findall(r"[a-zA-Z]+", text))
  clear_text  = Preprocessing_AR.process(arabic)+Preprocessing_EN.process(english)
  return clear_text

class Preprocessing_EN:
  @staticmethod
  def process(sentance):
    sentance = Preprocessing_EN.remove_punctuation(sentance)
    sentance = Preprocessing_EN.tokenizer(sentance)
    sentance = Preprocessing_EN.normalizer(sentance)
    sentance = Preprocessing_EN.remove_stopwords(sentance)
    sentance = Preprocessing_EN.remove_deplicate(sentance)
    sentance = Preprocessing_EN.stemmer(sentance)
    return sentance

  @staticmethod
  def tokenizer(sentance):
    words = word_tokenize(sentance)
    return words

  @staticmethod
  def normalizer(sentance):
    return [word.lower() for word in sentance]

  @staticmethod
  def remove_stopwords(sentance):
    stop_words = set(stopwords.words('english'))
    sentance = [word for word in sentance if word not in stop_words]
    return sentance
  
  @staticmethod
  def stemmer(sentance):
    stemmer = PorterStemmer()
    sentance = [stemmer.stem(word) for word in sentance]
    return sentance

  @staticmethod
  def remove_punctuation(sentance):
    return re.sub(r'[^A-Za-z\s]',' ',sentance)

  @staticmethod
  def remove_deplicate(sentance):
    return list(set(sentance))
  

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

class Preprocessing_AR:
    @staticmethod
    def process(sentence):
        # Basic preprocessing steps
        tokens = Preprocessing_AR.remove_punctuation(sentence)
        tokens = Preprocessing_AR.tokenizer(tokens)
        tokens = Preprocessing_AR.normalizer(tokens)
        tokens = Preprocessing_AR.remove_stopwords(tokens)
        tokens = Preprocessing_AR.stemmer(tokens)
        return tokens

    @staticmethod
    def tokenizer(sentence):
        return word_tokenize(sentence)

    @staticmethod
    def normalizer(tokens):
        # Normalize each token
        return [
            re.sub(r'[\u064B-\u065F]', '', token)  # Remove diacritics
            .replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')  # Normalize Alef
            .replace('ى', 'ي').replace('ؤ', 'و').replace('ئ', 'ي')  # Normalize other letters
            for token in tokens
        ]

    @staticmethod
    def remove_stopwords(tokens):
        stop_words = set(stopwords.words('arabic'))
        return [word for word in tokens if word not in stop_words]

    @staticmethod
    def stemmer(tokens):
        ArListem = ArabicLightStemmer()
        # Apply light stemming and filter out overly short stems
        return [
            ArListem.light_stem(token) or token  # Fallback to the original word if stemming fails
            for token in tokens
            if len(token) >= 3  # Remove stems shorter than 3 characters
        ]

    @staticmethod
    def remove_punctuation(sentence):
        # Replace non-Arabic letters, numbers, or spaces with a space
        return re.sub(r'[^ا-ي0-9\s]', ' ', sentence)



def compute_similarity(df, query, text_column="Text", top_n=8):

    restult = []
    # Drop NaN values in the text column
    df = df.dropna(subset=[text_column])
    
    # Convert text data to a list
    texts = df[text_column].astype(str).tolist()
    
    # Add the query to the list
    texts.append(query)
    
    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Compute cosine similarity between the query and all texts
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Get top N similar texts
    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    # Return results with similarity scores
    return df.iloc[top_indices].assign(Similarity_Score=similarity_scores[top_indices])['lawyer_id']


def convert2json(data):
    # Convert to JSON format
    result = {
        "isSuccess": True,
        "lawyers": data.to_dict(orient="records")  # Convert DataFrame to list of dictionaries
    }
    return result