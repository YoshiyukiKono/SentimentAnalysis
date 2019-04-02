from collections import Counter

#words = []
#for tokens in tokenized:
#    for token in tokens:
#        words.append(token)
out_list = tokenized
words = [element for in_list in out_list for element in in_list]

print(words[:13])
print(len(words))

"""
Create a vocabulary by using Bag of words
"""

# TODO: Implement 

word_counts = Counter(words)
sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
vocab_to_int = {word:ii for ii, word in int_to_vocab.items()}

print("len(sorted_vocab):",len(sorted_vocab))
print("sorted_vocab - top:", sorted_vocab[:3])
print("sorted_vocab - least:", sorted_vocab[-15:])

# Dictionart that contains the Frequency of words appearing in messages.
# The key is the token and the value is the frequency of that word in the corpus.
total_count = len(words)
freqs = {word: count/total_count for word, count in word_counts.items()}

print("freqs[supplication]:",freqs["supplication"] )
print("freqs[the]:",freqs["the"] )