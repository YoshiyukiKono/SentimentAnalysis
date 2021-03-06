import pickle
import re
import nltk
import numpy as np

#with open('vocab.pickle', 'rb') as f:
#    vocab_l = pickle.load(f)

nltk.download('wordnet')

vocab_l = pickle.load(open('vocab.pickle', 'rb'))

import torch

from sentiment import TextClassifier


PATH = "./model.torch"
# torch.load with map_location='cpu'
model_l = torch.load(PATH,map_location='cpu')
#model_l.to("cpu")

class UnknownWordsError(Exception):
  "Only unknown words are included in text"

def preprocess(message):
    """
    This function takes a string as input, then performs these operations: 
        - lowercase
        - remove URLs
        - remove ticker symbols 
        - removes punctuation
        - tokenize by splitting the string on whitespace 
        - removes any single character tokens
    
    Parameters
    ----------
        message : The text message to be preprocessed.
        
    Returns
    -------
        tokens: The preprocessed text into tokens.
    """ 
    #TODO: Implement 
    
    # Lowercase the twit message
    text = message.lower()
    
    # Replace URLs with a space in the message
    text = re.sub("http(s)?://([\w\-]+\.)+[\w-]+(/[\w\- ./?%&=]*)?",' ', text)
    
    # Replace ticker symbols with a space. The ticker symbols are any stock symbol that starts with $.
    text = re.sub("\$[^ \t\n\r\f]+", ' ', text)
    
    # Replace StockTwits usernames with a space. The usernames are any word that starts with @.
    text = re.sub("@[^ \t\n\r\f]+", ' ', text)

    # Replace everything not a letter with a space
    text = re.sub("[^a-z]", ' ', text)
    
    
    # Tokenize by splitting the string on whitespace into a list of words
    tokens = text.split()

    # Lemmatize words using the WordNetLemmatizer. You can ignore any word that is not longer than one character.
    wnl = nltk.stem.WordNetLemmatizer()
    tokens = [wnl.lemmatize(w, pos='v') for w in tokens if len(w) > 1]
    
    return tokens


def predict_func(text, model, vocab):
    """ 
    Make a prediction on a single sentence.

    Parameters
    ----------
        text : The string to make a prediction on.
        model : The model to use for making the prediction.
        vocab : Dictionary for word to word ids. The key is the word and the value is the word id.

    Returns
    -------
        pred : Prediction vector
    """
    
    # TODO Implement
    tokens = preprocess(text)
    

    # Filter non-vocab words
    tokens = [token for token in tokens if token in vocab] #pass
    # Convert words to ids
    tokens = [vocab[token] for token in tokens] #pass

    if len(tokens) == 0:
      raise UnknownWordsError

    # Adding a batch dimension
    text_input = torch.from_numpy(np.asarray(torch.LongTensor(tokens).view(-1, 1)))

    # Get the NN output       
    batch_size = 1
    hidden = model.init_hidden(batch_size) #pass
    
    logps, _ = model(text_input, hidden) #pass
    # Take the exponent of the NN output to get a range of 0 to 1 for each label.
    pred = torch.round(logps.squeeze())#pass
    pred = torch.exp(logps) 
    
    return pred


def predict_api(args):
  text = args.get('text')
  try:
    result = predict_func(text, model_l, vocab_l)
    return result.detach().numpy()[0]
  except UnknownWordsError:
    return [0,0,1,0,0]
    

#"message": "The result could not be encoded in JSON.",
args = {"text": "Google is working on self driving cars, I'm bullish on $goog"}
#args = {"text": "I'm bullish on $goog"}
args = {"text": "I'll strongly recommend to buy on $goog"}
#args = {"text": "日本語だとどうなる $goog"}
#args = {"text": "elyoq baoq pquq $goog"}
result = predict_api(args)
print(result)
