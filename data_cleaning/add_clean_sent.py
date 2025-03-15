from deep_translator import GoogleTranslator
from pymongo import MongoClient
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# MongoDB connection details
mongo_uri = "mongodb://localhost:27017/"
database_name = "twitter_database"
collection_name = "tweets_collection"  # Adjust collection name as needed


# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to translate text to English
def translate_to_english(text, lang):
    translated_text = GoogleTranslator(source='auto', target='en').translate(text)
    return translated_text

# Function to clean text by lemmatizing
def clean_text(text):
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return lemmatized_text.strip()

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    compound_score = sentiment['compound']
    return compound_score

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client[database_name]
collection = db[collection_name]

# Get the total number of documents
total_docs = collection.count_documents({})

# Process each document in the collection with a progress bar
for doc in tqdm(collection.find(), total=total_docs, desc="Processing documents"):
    try:
        # Extract necessary fields
        text = doc.get("text")
        lang = doc.get("lang")
        
        # Translate text if necessary
        if lang != "en":
            text = translate_to_english(text, lang)
        
        # Clean text (if additional cleaning is needed)
        text = clean_text(text)
        
        # Analyze sentiment
        sentiment_score = analyze_sentiment(text)
        
        # Update document in MongoDB with sentiment score
        collection.update_one(
            {"_id": doc["_id"]}, 
            {"$set": {"sentiment_score": sentiment_score}}
        )
        
        # print(f"Sentiment score added to document {doc['_id']}")
        
    except Exception as e:
        print(f"Error processing document {doc['_id']}: {str(e)}")
