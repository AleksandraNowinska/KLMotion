
import json
import pandas as pd
from datetime import datetime

# Define airline user IDs
airline_user_id_lst = [56377143, 106062176, 124476322, 18332190, 22536055, 20626359]

def find_initial_tweet_and_first_reply(tweet, airline_user_ids):
    initial_tweet = None
    first_airline_reply = None
    tweet_queue = [(tweet, None)]

    while tweet_queue:
        current_tweet, parent_tweet = tweet_queue.pop(0)
        if current_tweet["user_id"] in airline_user_ids:
            if parent_tweet:
                first_airline_reply = current_tweet
                initial_tweet = parent_tweet
                break
        if "replies" in current_tweet:
            for reply in current_tweet["replies"]:
                tweet_queue.append((reply, current_tweet))
    
    if initial_tweet and first_airline_reply:
        return (
            initial_tweet["user_id"],
            initial_tweet.get("sentiment_score", None),
            first_airline_reply["user_screen_name"],
            initial_tweet["created_at"],
            first_airline_reply["created_at"],
            initial_tweet["lang"],
            len(initial_tweet['text'])
        )
    return (None, None, None, None, None, None, None)

def find_last_instance(tweet, user_id, airline_user_ids):
    last_instance = None
    tweet_queue = [(tweet, None)]
    tweet_count = 0
    user_tweet_index = None
    airline_reply_index = None

    while tweet_queue:
        current_tweet, parent_tweet = tweet_queue.pop(0)
        if current_tweet["user_id"] == user_id:
            tweet_count += 1
            if parent_tweet and parent_tweet["user_id"] in airline_user_ids:
                last_instance = current_tweet
                if user_tweet_index is None:
                    user_tweet_index = tweet_count
                airline_reply_index = tweet_count
        if "replies" in current_tweet:
            for reply in current_tweet["replies"]:
                tweet_queue.append((reply, current_tweet))
    
    if last_instance:
        tweets_between = airline_reply_index - user_tweet_index + 1
    else:
        tweets_between = 0
    
    return last_instance, tweets_between

def calculate_time_difference_hour(start_time_str, end_time_str):
    if start_time_str and end_time_str:
        start_time = datetime.strptime(start_time_str, "%a %b %d %H:%M:%S %z %Y")
        end_time = datetime.strptime(end_time_str, "%a %b %d %H:%M:%S %z %Y")
        time_diff = end_time - start_time
        return time_diff.total_seconds() / 3600  # Convert seconds to hours
    return None

def calculate_time_difference_minute(start_time_str, end_time_str):
    if start_time_str and end_time_str:
        start_time = datetime.strptime(start_time_str, "%a %b %d %H:%M:%S %z %Y")
        end_time = datetime.strptime(end_time_str, "%a %b %d %H:%M:%S %z %Y")
        time_diff = end_time - start_time
        return time_diff.total_seconds() / 60  # Convert seconds to minutes
    return None

def extract_date_components(date_str):
    if date_str:
        date_obj = datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
        return {
            'hour': date_obj.hour,
            'day_of_week': date_obj.strftime("%A"),
            'month': date_obj.strftime("%B")
        }
    return {
        'hour': None,
        'day_of_week': None,
        'month': None
    }

def parse_datetime(date_str):
    return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")


sent_evo_list = []

# Load the JSON data from file and process each line
with open('path_to_convo_add_sentiment_output', 'r', encoding='utf-8') as all_convo:
    for convo in all_convo:
        try:
            conversation = json.loads(convo)

            (
                user_tweet_user_id,
                initial_sentiment,
                airline_user_screen_name,
                user_tweet_created_at,
                airline_tweet_created_at,
                user_tweet_lang,
                inital_length
            ) = find_initial_tweet_and_first_reply(conversation, airline_user_id_lst)

            if user_tweet_user_id is not None and airline_user_screen_name is not None:
                last_instance, tweets_between = find_last_instance(conversation, user_tweet_user_id, airline_user_id_lst)

                if last_instance is not None:
                    sentiment_change = last_instance.get("sentiment_score", 0) - initial_sentiment
                    time_diff_hours = calculate_time_difference_hour(user_tweet_created_at, airline_tweet_created_at)
                    time_diff_mins = calculate_time_difference_minute(user_tweet_created_at, airline_tweet_created_at)

                    user_tweet_date_components = extract_date_components(user_tweet_created_at)
                    airline_tweet_date_components = extract_date_components(airline_tweet_created_at)

                    user_tweet_datetime = parse_datetime(user_tweet_created_at)
                    airline_tweet_datetime = parse_datetime(airline_tweet_created_at)

                
                    sent_evo_dict = {
                        'airline': airline_user_screen_name,
                        'customer_id': user_tweet_user_id,
                        'initial_cust_sent': initial_sentiment,
                        'final_cust_sent': last_instance.get("sentiment_score", 0),
                        'first_user_tweet_date': user_tweet_datetime,
                        'airline_response_date': airline_tweet_datetime,
                        'lang': user_tweet_lang,
                        'inital_length': inital_length,
                        'sentiment_change': sentiment_change,
                        'response_time_hours': time_diff_hours,
                        'response_time_minutes': time_diff_mins,
                        'user_tweet_hour': user_tweet_date_components['hour'],
                        'user_tweet_day_of_week': user_tweet_date_components['day_of_week'],
                        'user_tweet_month': user_tweet_date_components['month'],
                        'airline_tweet_hour': airline_tweet_date_components['hour'],
                        'airline_tweet_day_of_week': airline_tweet_date_components['day_of_week'],
                        'airline_tweet_month': airline_tweet_date_components['month'],
                        'tweets_between': tweets_between
                    }
                    sent_evo_list.append(sent_evo_dict)
                else:
                    print(f"No matching last instance found for user_id: {user_tweet_user_id}")
            else:
                pass #print(f"No airline reply found in conversation.")
                
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except KeyError as e:
            print(f"KeyError: {e} - 'tweet' key not found in JSON entry.")
        except TypeError as e:
            print(f"TypeError: {e} - Expected dictionary but got {type(conversation)}")
        except Exception as e:
            print(f"Unexpected error: {e}")


# Create DataFrame
sentiment_evolution_df = pd.DataFrame(sent_evo_list)

# Export DataFrame to CSV
sentiment_evolution_df.to_csv("sentiment_evolution.csv", index=False)
