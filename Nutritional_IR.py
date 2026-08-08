import PyPDF2
import string
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

pdf = [
    r"C:\Users\harivarman\Desktop\nutritional IR\Source file\healthy-diet-fact-sheet-394.pdf"
]

documents = []

for file in pdf:
    reader = PyPDF2.PdfReader(file)

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    documents.append(text)


def preprocessing(sentences):

    clean_words = []

    for sent in sentences:

        words = word_tokenize(sent)

        for word in words:

            word = word.lower()

            if word in string.punctuation:
                continue

            if word in stop_words:
                continue

            stem = stemmer.stem(word)

            clean_words.append(stem)

    return " ".join(clean_words)


cleaned_documents = []

for doc in documents:

    sentences = sent_tokenize(doc)

    cleaned = preprocessing(sentences)

    cleaned_documents.append(cleaned)

    print(cleaned)


vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(cleaned_documents)

user = input("Enter Your Nutritional Query Here : ")

user_clean = preprocessing([user])

user_matrix = vectorizer.transform([user_clean])

similarity = cosine_similarity(tfidf_matrix, user_matrix)

top = np.argmax(similarity)

best_doc = documents[top]

sentences = sent_tokenize(best_doc)

cleaned_sentences = []

original_sentences = []

for s in sentences:

    cleaned = preprocessing([s])

    if cleaned.strip() == "":
        continue

    cleaned_sentences.append(cleaned)
    original_sentences.append(s)

result_vectorizer = TfidfVectorizer()

result_matrix = result_vectorizer.fit_transform(cleaned_sentences)

user_result = result_vectorizer.transform([user_clean])

similarity = cosine_similarity(result_matrix, user_result)

best_result = np.argmax(similarity)

print("Here's your information:\n")
print(original_sentences[best_result])