### I will solve the given project with the lstm that is long short term memory.
##### importing the required libraries for building the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
#### importing the libraries for scrapping the data from the url and then parsing the data
##pip install requests
##pip install html5lib
###pip install bs4
from bs4 import BeautifulSoup as bs
import requests as re
#### here i will scrap the data from different sites the domain will be related to data science and ai
# url
url="https://insights.blackcoffer.com/ai-in-healthcare-to-improve-patient-outcomes/"
# get the HTML
r=re.get(url)
htmlcontent=r.content
print(htmlcontent)
# parsing the html
soup=bs(htmlcontent,"html.parser")
print(soup)
print(soup.prettify)
# getting all the paras
paras=soup.findAll("p")
print(paras)
# now getting the required text
text=soup.findAll(attrs={"class":"td-post-content"})
print(text)
#now removing the unwanted symbols and unrequired text
text_new=text[0].text
print(text_new)
####now scrapping more websites to get more data
# url
url2="https://insights.blackcoffer.com/ai-human-robotics-machine-future-planet-blackcoffer-thinking-jobs-workplace/"
# get the HTML
r2=re.get(url)
htmlcontent2=r2.content
print(htmlcontent2)
# parsing the html

# getting all the paras
paras2=soup.findAll("p")
print(paras2)
# now getting the required text
text2=soup.findAll(attrs={"class":"td-post-content"})
print(text2)
#now removing the unwanted symbols and unrequired text
text_new2=text2[0].text
print(text_new2)
# url
url3="https://insights.blackcoffer.com/deep-learning-impact-on-areas-of-e-learning/"
# get the HTML
r3=re.get(url)
htmlcontent3=r3.content
print(htmlcontent3)
# parsing the html

# getting all the paras
paras3=soup.findAll("p")
print(paras3)
# now getting the required text
text3=soup.findAll(attrs={"class":"td-post-content"})
print(text3)
#now removing the unwanted symbols and unrequired text
text_new3=text3[0].text
print(text_new3)
# url
url4="https://insights.blackcoffer.com/how-data-analytics-can-help-your-business-respond-to-the-impact-of-covid-19/"
# get the HTML
r4=re.get(url)
htmlcontent4=r4.content
print(htmlcontent3)
# parsing the html

# getting all the paras
paras4=soup.findAll("p")
print(paras4)
# now getting the required text
text4=soup.findAll(attrs={"class":"td-post-content"})
print(text4)
#now removing the unwanted symbols and unrequired text
text_new4=text4[0].text
print(text_new4)
#### now concating all the paras as one and taking it as corpus
corpus=text_new+text_new2+text_new3+text_new4
print(corpus)
final_corpus=[corpus]
type(final_corpus)
### so now we have created a dataset related to machine learning and ai which is the final_corpus now we should move forward to build a text generative model using lstm####
##### creating the input output pair for training the model.
tokenizer = tf.keras.preprocessing.text.Tokenizer()
##### tokenizing the final_corpus
tokenizer.fit_on_texts(final_corpus)
sequences = tokenizer.texts_to_sequences(final_corpus)
print(sequences)

vocab_size = len(tokenizer.word_index) + 1
input_sequences = []
output_words = []
sequence_length = 20
for sequence in sequences:
    for i in range(sequence_length, len(sequence)):
        input_sequences.append(sequence[i - sequence_length:i])
        output_words.append(sequence[i])

input_sequences = np.array(input_sequences)
output_words = np.array(output_words)

# Reshape input sequences to have 3 dimensions
input_sequences = np.reshape(input_sequences, (input_sequences.shape[0], sequence_length, 1))

model = Sequential()
model.add(LSTM(200, input_shape=(sequence_length, 1)))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.fit(input_sequences, output_words, epochs=50)

def generate_sentence(seed_text, num_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=sequence_length)
    predicted_words = []

    for _ in range(num_words):
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = tokenizer.index_word[predicted[0]]
        predicted_words.append(output_word)
        token_list = np.append(token_list[:, 1:], np.expand_dims(predicted, axis=1), axis=1)

    generated_sentence = seed_text + " " + " ".join(predicted_words)
    return generated_sentence

seed_text = "If anything kills over 10 million people in the next few decades, it will be"
num_words = 200

generated_sentence = generate_sentence(seed_text, num_words)
print(generated_sentence)



##### we can also save the above model in pkl format so that it can be deployed using the flask
#### so saving the above model in the pkl format
import pickle#### import pickle
# Save the LSTM model to a  pkl file
with open("lstm_model.pkl", "wb") as file:
    pickle.dump(model, file)

















