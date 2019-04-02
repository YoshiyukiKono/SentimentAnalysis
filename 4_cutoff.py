# Float that is the frequency cutoff. Drop words with a frequency that is lower or equal to this number.
low_cutoff = 0.000002

# Integer that is the cut off for most common words. Drop words that are the `high_cutoff` most common words.
"""
example_count = []
example_count.append(sorted_vocab.index("the"))
example_count.append(sorted_vocab.index("for"))
example_count.append(sorted_vocab.index("of"))
print(example_count)
high_cutoff = min(example_count)
"""
high_cutoff = 20
print("high_cutoff:",high_cutoff)
print("low_cutoff:",low_cutoff)

# The k most common words in the corpus. Use `high_cutoff` as the k.
#K_most_common = [word for word in sorted_vocab[:high_cutoff]]
K_most_common = sorted_vocab[:high_cutoff]

print("K_most_common:",K_most_common)


## Updating Vocabulary by Removing Filtered Words

filtered_words = [word for word in freqs if (freqs[word] > low_cutoff and word not in K_most_common)]

print("len(filtered_words):",len(filtered_words))

# A dictionary for the `filtered_words`. The key is the word and value is an id that represents the word. 
vocab =  {word:ii for ii, word in enumerate(filtered_words)}
# Reverse of the `vocab` dictionary. The key is word id and value is the word. 
id2vocab = {ii:word for word, ii in vocab.items()}
# tokenized with the words not in `filtered_words` removed.

print("len(tokenized):", len(tokenized))

filtered = [[token for token in tokens if token in vocab] for tokens in tokenized]
print("len(filtered):", len(filtered))
print("tokenized[:1]", tokenized[:1])
print("filtered[:1]",filtered[:1])
