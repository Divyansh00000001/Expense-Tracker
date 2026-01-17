# api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json
from rest_framework.permissions import IsAuthenticated 
from .serializers import YourDataSerializer  


nltk.download('punkt')
nltk.download('stopwords')

# A small keyword-based mapping to catch common items that the ML model may misclassify.
# Add or tweak keywords to match your project's Category names.
KEYWORD_CATEGORY_MAP = {
    'soap': 'Household',
    'shampoo': 'Household',
    'detergent': 'Household',
    'toothpaste': 'Household',
    'razor': 'Personal Care',
    'shaver': 'Personal Care',
    'blade': 'Personal Care',
    'trimmer': 'Personal Care',
    'bread': 'Food',
    'milk': 'Food',
    'rice': 'Food',
    'vegetable': 'Food',
    'groceries': 'Food',
    'bus': 'Transport',
    'uber': 'Transport',
    'ola': 'Transport',
    'rapido': 'Transport',
    'car': 'Transport',
    'bike': 'Transport',
    'tshirt': 'Clothing',
    't': 'Clothing',
    'shirt': 'Clothing',
    'pajamas': 'Clothing',
    'pajama': 'Clothing',
    'pants': 'Clothing',
    'jeans': 'Clothing',
    'toy': 'Entertainment',
    'insurance': 'Insurance',
    'emi': 'EMI',
    'headphones': 'Electronics',
}


def preprocess_text(text):
    """Normalize and tokenize text for both dataset preprocessing and incoming requests."""
    if not isinstance(text, str):
        return ''
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    return ' '.join(tokens)


# Initialize a cached TF-IDF vectorizer and model at import time to improve
# performance and consistency. If dataset or resources are unavailable,
# fall back to None and attempt on-demand training later.
try:
    data = pd.read_csv('dataset.csv')
    if 'clean_description' not in data.columns:
        data['clean_description'] = data['description'].apply(preprocess_text)
    else:
        data['clean_description'] = data['clean_description'].fillna('').astype(str)

    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(data['clean_description'])
    model = RandomForestClassifier()
    model.fit(X, data['category'])
except Exception as e:
    tfidf_vectorizer = None
    model = None
    print('Warning: failed to initialize category model at import time:', e)


def map_to_existing_category(name):
    """If a Category object exists with the given name (case-insensitive),
    return the canonical name from the DB. Otherwise return the input name.
    This helps ensure predicted categories match what your `Category` model uses.
    """
    if not name:
        return name
    try:
        from expenses.models import Category
        cat = Category.objects.filter(name__iexact=name).first()
        if cat:
            return cat.name
    except Exception:
        # If DB access fails (e.g., during migrations), just return the name
        pass
    return name


class PredictCategory(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user_input = request.data.get('description')
        # quick keyword-based override
        def keyword_lookup(text):
            if not text:
                return None
            tokens = preprocess_text(text).split()
            for t in tokens:
                if t in KEYWORD_CATEGORY_MAP:
                    return KEYWORD_CATEGORY_MAP[t]
            return None

        kw_category = keyword_lookup(user_input)
        if kw_category:
            mapped = map_to_existing_category(kw_category)
            return Response({'predicted_category': mapped}, status=status.HTTP_200_OK)

        # If we have a cached model and vectorizer, use them. Otherwise try
        # a quick on-demand training as a fallback.
        if tfidf_vectorizer is None or model is None:
            try:
                data = pd.read_csv('dataset.csv')
                data['clean_description'] = data['description'].apply(preprocess_text)
                local_tfidf = TfidfVectorizer()
                local_X = local_tfidf.fit_transform(data['clean_description'])
                local_model = RandomForestClassifier()
                local_model.fit(local_X, data['category'])
                user_input = preprocess_text(user_input)
                user_input_vector = local_tfidf.transform([user_input])
                predicted_category = local_model.predict(user_input_vector)
                mapped = map_to_existing_category(predicted_category[0])
                return Response({'predicted_category': mapped}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({'predicted_category': 'Others'}, status=status.HTTP_200_OK)

        try:
            user_input = preprocess_text(user_input)
            user_input_vector = tfidf_vectorizer.transform([user_input])
            predicted_category = model.predict(user_input_vector)
            mapped = map_to_existing_category(predicted_category[0])
            return Response({'predicted_category': mapped}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'predicted_category': 'Others'}, status=status.HTTP_200_OK)




class UpdateDataset(APIView):
    # permission_classes = [IsAuthenticated]

    def post(self, request):
       new_data = request.data.get('new_data')

       if 'description' in new_data and 'category' in new_data:
            # Load your existing dataset
            data = pd.read_csv('dataset.csv')  # Load the existing dataset
            new_category = new_data['category']
            new_description = new_data['description']

            # Append the new data to the dataset
            new_row = {'description': new_description, 'category': new_category, 'clean_description': preprocess_text(new_description)}
            data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
            # Save the updated dataset
            data.to_csv('dataset.csv', index=False)

            # Retrain the module-level (cached) model and vectorizer so future
            # predict requests use the updated dataset without re-reading the file.
            try:
                global tfidf_vectorizer, model
                tfidf_vectorizer = TfidfVectorizer()
                X = tfidf_vectorizer.fit_transform(data['clean_description'])
                model = RandomForestClassifier()
                model.fit(X, data['category'])
            except Exception as e:
                print('Warning: failed to retrain cached model after dataset update:', e)
