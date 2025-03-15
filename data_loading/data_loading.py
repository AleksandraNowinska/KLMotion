import os
import json
import pymongo
from pymongo import MongoClient
 
# Configure MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['twitter_db']
collection = db['tweets']
 
# Directory containing the JSON files
directory_path = 'path_to_twitter_data'
 
# Function to insert JSON objects from a file into MongoDB
def insert_file(file_path):
    try:
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                if line.strip():  # Skip empty lines
                    try:
                        json_object = json.loads(line)
                        document = {
                            'file_name': os.path.basename(file_path),
                            'json_data': json_object
                        }
                        collection.insert_one(document)
                    except json.JSONDecodeError as e:
                        print(f"JSONDecodeError: Error reading JSON data from file {file_path}, line {line_number}: {e}")
    except IOError as e:
        print(f"IOError: Error reading file {file_path}: {e}")
    except pymongo.errors.PyMongoError as e:
        print(f"PyMongoError: Error uploading data from file {file_path} to MongoDB: {e}")
    except Exception as e:
        print(f"Unexpected error with file {file_path}: {e}")
 
# Iterate over files in the directory and insert them into MongoDB
file_count = 0
error_count = 0
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if os.path.isfile(file_path) and filename.endswith('.json'):
        try:
            insert_file(file_path)
            file_count += 1
        except Exception as e:
            print(f"Unexpected error while processing file {file_path}: {e}")
            error_count += 1
 
print(f"All {file_count} files have been processed.")
if error_count > 0:
    print(f"There were {error_count} files that encountered errors.")