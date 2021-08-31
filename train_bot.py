import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer= WordNetLemmatizer()
import json
import pickle
# nltk.download()
words =[]
classes =[]
documents=[]
ignore_words=['?','!','.','-']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenization
        w=nltk.word_tokenize(pattern)
        # print('Token is :{}'.format((w)))
        words.extend(w)
        documents.append((w,intent['tag']))
    #     add the tag to classes list
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

    #     final lists
    # print('Words list is: {}'.format(words))
    # print('Docs are: {}'.format(documents))
    # print('Classes are: {}'.format(classes))

words=[lemmatizer.lemmatize(w.lower())for w in words if w not in ignore_words]
words=list(set(words))
classes=list(set(classes))
# print(words)
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

