# This is conversations without sentiment so it's significantly faster than the other!!

from pymongo import MongoClient
from collections import defaultdict
import json

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['twitter_database']
tweets_collection = db['tweets']
print("Connected to MongoDB database:", db.name)


twt_cnt = 0
tweet_replies = {} # for every tweet_id it says which tweet this is replying to
reply_count_dict = defaultdict(int) # For every tweet it says how many replies it has if it has > 0
tweet_dict = {0:{'tweet_id': 0, 'trxt': 'that weird', 'replies': []}} # dictionary with all valid tweets
all_tweets = tweets_collection.find() # importing all tweets

print('Start tweet import')
# Import all tweets from MongoDB
for tweet_doc in all_tweets:
    tweet = tweet_doc.get('json_data', {})  # Ensure tweet is not None
    if not tweet:
        continue  # Skip this tweet if it's None
    tweet_id = tweet.get('id')
    try:
        user_id = tweet.get('user').get('id')
    except:
        continue
    try:
        created_at = tweet.get('created_at')
    except:
        continue
    if not tweet or tweet_id in tweet_dict.keys():
        continue  # Skip this tweet if it's None, missing an ID, or already processed
    # If there is a full text, this will be the text, otherwise text will be the text
    try:
        tweet_text = tweet.get('extended_tweet').get('full_text') 
    except:
        tweet_text = tweet.get('text')
    try:
        entities = tweet.get('entities', {})
        user_mentions_object = entities.get('user_mentions', [])
        user_mentions = [mention.get('id') for mention in user_mentions_object]
    except:
        user_mentions = []

    tweet_lang = tweet.get('lang')
    user_scrn_name = tweet.get('user').get('screen_name')
    
    # We store all tweets as tweet dictionaries, with a few fields we are interested in
    tweet_dict[tweet_id] = {
        'created_at': created_at,
        'tweet_id': tweet_id,
        'user_id': user_id,
        'lang': tweet_lang,
        'user_mentions': user_mentions,
        'replies': [],
        'user_screen_name': user_scrn_name,
        'text': tweet_text
    }
    reply_to = tweet.get('in_reply_to_status_id')
    # All the tweets that are a reply to something get a dict with the value being the tweet that it is replying to
    if reply_to:
        tweet_replies[tweet_id] = reply_to
        reply_count_dict[reply_to] += 1
    twt_cnt += 1
    print('tweets added to tweet_dict:', twt_cnt)
print('all tweets added to tweet_dict!')

def add_origin_twt(start_tweet):
    """
    Add a top-level / root / parent / origin tweet to the tree dictionary.
    """
    tree_dict[start_tweet['tweet_id']] = start_tweet

def add_replies(dictionary, new_replies):
    if dictionary:
        if 'replies' not in dictionary:
            dictionary['replies'] = []

        # Check if the 'replies' list is empty
        if not dictionary:
            # If empty, append the new 'replies' section directly
            dictionary.append(new_replies)
            return
        
        # Get the last 'replies' section
        current_level = dictionary
        while current_level and 'replies' in current_level[-1]:
            current_level = current_level[-1]['replies']
        
        # Append the new 'replies' section
        current_level.append(new_replies)

def add_reply_srch(tree_dict, tweet_id, reply_tweet):
    """
    Add a reply to a specific tweet identified by tweet_id.
    """
    replied_to = find_tweet_by_id(tree_dict, tweet_id)
    if replied_to:
        if 'replies' not in replied_to:
            replied_to['replies'] = []
        replied_to['replies'].append(reply_tweet)
    else:
        print(f"Tweet with id {tweet_id} not found")
        raise ValueError(f"Tweet with id {tweet_id} not found") # This should never happen


def add_reply_know(known_dict, reply_tweet):
    """
    Add a reply to a specific tweet identified by a known dict.
    """
    replied_to = known_dict
    if 'replies' not in replied_to:
        replied_to['replies'] = []
    replied_to['replies'].append(reply_tweet)

def find_tweet_by_id(tree_dict, tweet_id):
    """
    Recursively find a tweet by its ID.
    """
    if tweet_id in tree_dict:
        return tree_dict[tweet_id]
    for tweet in tree_dict.values():
        if 'replies' in tweet:
            for reply in tweet['replies']:
                found = find_tweet_by_id({reply['tweet_id']: reply}, tweet_id)
                if found:
                    return found
    return None  # Return None if tweet not found

def find_tweet_by_path(tree_dict, id_path_list, path_index):
    """
    Recursively find a tweet by following a path of IDs.
    
    Args:
        tree_dict (dict): The dictionary representing the tweet tree.
        id_path (list): The list of IDs representing the path to the desired tweet.
        
    Returns:
        dict or None: The tweet dictionary if found, otherwise None.
    """
    id_path = id_path_list[:path_index+1]
    if not id_path:
        return None
    
    current_id = id_path[0]
    if current_id in tree_dict:
        tweet = tree_dict[current_id]
        if len(id_path) == 1:
            return tweet
        if 'replies' in tweet:
            reply_dict = {reply['tweet_id']: reply for reply in tweet['replies']}
            return find_tweet_by_path(reply_dict, id_path[1:], path_index)
    
    return "can't find it mate"

airline_user_id_lst = [56377143, 106062176, 18332190, 22536055, 124476322, 26223583, 2182373406, 38676903, 1542862735, 253340062, 218730857, 45621423, 20626359]
tree_dict = {}

sorted_reply_count_dict =  dict(sorted(reply_count_dict.items(), key=lambda item: item[1]))

print('Starting construction of conversations..')
# Loops over every tweet that has at least 1 reply, starting from the lowest
for reply_cnt in sorted_reply_count_dict:
    tweet_id = reply_cnt
    try:
        in_reply_to_twt = tweet_replies[tweet_id] 
    except:
        pass 
    convo_line_lst = []

    try: 
        convo_line_lst.append(tweet_dict[tweet_id])
    except:
        continue
    # We add the whole conversation chain up till the original tweet to this list
    while in_reply_to_twt in tweet_replies: 
        convo_line_lst.append(tweet_dict[in_reply_to_twt])
        reply_count_dict[in_reply_to_twt] -= 1
        if reply_count_dict[in_reply_to_twt] < 1:
            del reply_count_dict[in_reply_to_twt]
        # If it cannot find the origin tweet, it will set the convo list to empty
        try:
            in_reply_to_twt = tweet_replies[in_reply_to_twt]
        except:
            convo_line_lst = []
            break
    # Check whether any of the list is already in the tree dictionary and if not add this
    if convo_line_lst != []:  
        if in_reply_to_twt in tweet_dict.keys():
            convo_line_lst.append(tweet_dict[in_reply_to_twt])
        else:
            pass 

        if convo_line_lst[-1]['user_id'] in airline_user_id_lst and len(convo_line_lst) > 2:
            rev_convo_line_lst = list(reversed(convo_line_lst))
            id_list = [tweet['tweet_id'] for tweet in rev_convo_line_lst]
            path_index = 0
            if isinstance(convo_line_lst[-1], dict) and convo_line_lst[-1]['tweet_id'] not in tree_dict: 
                add_origin_twt(convo_line_lst[-1])
            else: 
                # input any tweet not in tree_dict in reversed order as a reply chain
                for tweet in rev_convo_line_lst:
                    if path_index < len(rev_convo_line_lst) and len(rev_convo_line_lst) > 1:
                        if find_tweet_by_path(tree_dict, id_list, path_index) != "can't find it mate":
                            path_index += 1
                        else:
                            add_reply_srch(tree_dict, rev_convo_line_lst[path_index-1]['tweet_id'], tweet)
                            path_index += 1
        if len(convo_line_lst) > 2:
            try:
                if convo_line_lst[-1]['user_mentions'][0] in airline_user_id_lst:
                    airline_tagged = True
            except:
                airline_tagged = False

            if airline_tagged == True:
                rev_convo_line_lst = list(reversed(convo_line_lst))
                id_list = [tweet['tweet_id'] for tweet in rev_convo_line_lst]
                path_index = 0
                if isinstance(convo_line_lst[-1], dict) and convo_line_lst[-1]['tweet_id'] not in tree_dict: 
                    add_origin_twt(convo_line_lst[-1])
                else: 
                    # input any tweet not in tree_dict in reversed order as a reply chain
                    for tweet in rev_convo_line_lst:
                        if path_index < len(rev_convo_line_lst) and len(rev_convo_line_lst) > 1:
                            if find_tweet_by_path(tree_dict, id_list, path_index) != "can't find it mate":
                                path_index += 1
                            else:
                                add_reply_srch(tree_dict, rev_convo_line_lst[path_index-1]['tweet_id'], tweet)
                                path_index += 1


try:
    # Open a file in write mode
    with open('uncleaned_convo_objects_without_sentiment.json', 'w') as file:
        # Iterate through the big dictionary
        for key, sub_dict in tree_dict.items():
            # Convert each sub-dictionary to a JSON object
            json_object = json.dumps(sub_dict)
            # Write the JSON object to the file
            file.write(json_object + '\n')
    print("Data successfully written to output.json")
except Exception as e:
    print(f"An error occurred: {e}")
print(len(tree_dict))

# Code does not always manage to make sure that there are a certain amount of replies somehow, 
# so this has to be cleaned later or debugged