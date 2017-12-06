import pandas as pd
df = pd.read_csv('./zumepizza.csv')

NAME = df['NAME'][0]
FOLLOWER = df['FOLLOWER'][0]
LIKE = df['LIKE'][0]
TOTAL_RETWEET = df['TOTAL RETWEET']
TOTAL_LIKE = df['TOTAL_LIKE']
TWEETS = df['TWEET']

print(NAME)
print(FOLLOWER)
print(LIKE)
print(TOTAL_LIKE)
