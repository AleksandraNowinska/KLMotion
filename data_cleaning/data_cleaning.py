from pymongo import MongoClient, errors
from tqdm import tqdm #Gets vey cool loading bar

mongo_uri = "mongodb://localhost:27017/"
database_name = "twitter_database"
source_collection_name = "tweets"
target_collection_name = "tweets_collection"

# Mapping of original fields to new field names
fields_to_extract = {
    "json_data.id": "tweet_id",
    "json_data.user.id": "user_id",
    "json_data.datetime": "datetime",
    "json_data.in_reply_to_status_id": "in_reply_to_status_id",
    "json_data.in_reply_to_user_id": "in_reply_to_user_id",
    "json_data.entities.user_mentions": "user_mentions",
    "json_data.lang": "lang",
    "json_data.user.location": "user_location"
}

def get_nested_field(data, field_path):
    keys = field_path.split('.')
    for key in keys:
        if isinstance(data, list):
            try:
                key = int(key)
                data = data[key]
            except (ValueError, IndexError):
                return None
        elif isinstance(data, dict):
            data = data.get(key)
        else:
            return None
        if data is None:
            return None
    return data

try:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[database_name]
    source_collection = db[source_collection_name]
    target_collection = db[target_collection_name]

    seen_tweet_ids = set()

    with client.start_session() as session:
        cursor = source_collection.find({}, no_cursor_timeout=True, session=session)
        total_docs = source_collection.count_documents({})  # Get the total number of documents

        for doc in tqdm(cursor, total=total_docs, desc="Processing documents"):
            try:
                new_doc = {new_name: get_nested_field(doc, original_name) for original_name, new_name in fields_to_extract.items()}
                new_doc["_id"] = doc["_id"]

                tweet_id = new_doc.get("tweet_id")
                if tweet_id is None or tweet_id in seen_tweet_ids:
                    continue

                seen_tweet_ids.add(tweet_id)

                # Check if extended_tweet.full_text is present, if not use json_data.text
                full_text = get_nested_field(doc, "json_data.extended_tweet.full_text")
                if full_text:
                    new_doc["text"] = full_text
                else:
                    new_doc["text"] = get_nested_field(doc, "json_data.text")

                # Insert the new document into the target collection
                target_collection.insert_one(new_doc)
            
            except errors.PyMongoError as e:
                print(f"An error occurred while processing document with _id {doc['_id']}: {e}")

        cursor.close()
        print("Data successfully transferred to new collection.")

    # Create an index on datetime in the target collection
    target_collection.create_index([("datetime", 1)])
    print("Index on datetime created successfully.")
    target_collection.create_index([("user_id", 1)])

except errors.ServerSelectionTimeoutError as err:
    print("Failed to connect to MongoDB server:", err)
except errors.PyMongoError as err:
    print("An error occurred while working with MongoDB:", err)
finally:
    client.close()
