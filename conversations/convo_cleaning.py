# Get rid of convos with no replies
import json

input_file = 'uncleaned_convo_objects_without_sentiment.json'
output_file = 'cleaned_convo_objects_without_sentiment_v1.json'

filtered_data = []

# Read the file line by line and process each JSON object
with open(input_file, 'r') as file:
    for line in file:
        try:
            # Parse the JSON object
            doc = json.loads(line)
            # Filter out documents where the first 'replies' is an empty list
            if doc.get('replies'):
                filtered_data.append(doc)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

# Save the filtered data back to a new file
with open(output_file, 'w') as file:
    for doc in filtered_data:
        file.write(json.dumps(doc) + '\n')
