import pickle
#from singer import Singer

#singer = Singer('Shanranran')

with open('vocab.pickle', 'wb') as f:
    pickle.dump(vocab, f)
