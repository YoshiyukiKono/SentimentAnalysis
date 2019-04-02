with open('twits_dumped.json', 'r') as f:
    twits = json.load(f)

print(twits['data'][:10])
print(len(twits['data']))

messages = [twit['message_body'] for twit in twits['data']]
# Since the sentiment scores are discrete, we'll scale the sentiments to 0 to 4 for use in our network
sentiments = [twit['sentiment'] + 2 for twit in twits['data']]
