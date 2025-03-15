import json
from deep_translator import GoogleTranslator
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    lemmatized_text = lemmatizer.lemmatize(text)
    return lemmatized_text.strip()


# Function to translate text to English
def translate_to_english(text, lang):
    if lang != 'en':
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
    else:
        translated_text = text
    return translated_text


# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    compound_score = sentiment['compound']
    return compound_score


# Recursive function to process replies
def process_replies(reply_data):
    reply_text = reply_data.get("text", "")
    reply_lang = reply_data.get("lang", "en")
    
    if reply_lang != "en":
        reply_text = translate_to_english(reply_text, reply_lang)
        
    reply_text = clean_text(reply_text)
    sentiment_score_reply = analyze_sentiment(reply_text)
    reply_data["sentiment_score"] = sentiment_score_reply
    
    for reply in reply_data.get("replies", []):
        process_replies(reply)

file_path = r'output_of_convo_cleaning'
updated_data = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            data = json.loads(line.strip())
            lang = data.get("lang", "en")
            text = data.get("text", "")
            replies = data.get("replies", [])
            
            if lang != "en":
                text = translate_to_english(text, lang)
            
            text = clean_text(text)
            sentiment_score_main_text = analyze_sentiment(text)
            data["sentiment_score"] = sentiment_score_main_text  # Add sentiment score to main tweet object
            
            for reply_data in replies:
                process_replies(reply_data)
            
            updated_data.append(data)
            
        except json.JSONDecodeError:
            print("Error: Invalid JSON format")
            continue

# Write the updated data back to the JSON file
with open(file_path, 'w', encoding='utf-8') as file:
    for item in updated_data:
        json.dump(item, file)
        file.write('\n')

print("Sentiment scores added to the JSON file.")