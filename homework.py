import requests
import re
import csv
import string
import nltk
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
from nltk import word_tokenize
import networkx as nx
import matplotlib.pyplot as plt
from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    NamesExtractor,

    Doc
)

# The URL of our search
myUrl = 'https://habr.com/ru/search/?q=%D0%BA%D1%80%D0%B8%D0%BF%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D1%84%D0%B8%D1%8F&target_type=posts&order=relevance'

# Fetch the text from the URL
response = requests.get(myUrl)

if response.status_code == 200:
    # Create BeautifulSoup object
    bsText = BeautifulSoup(response.text, "html.parser")

    # Get text from the soup
    text = bsText.get_text()

    # Saving the text in .txt file
    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write(text)

# Saving the result in .txt file
f = open('output.txt', "r", encoding="utf-8")
textF = f.read()

text = textF.lower()

# Defining the special characters
spec_chars = string.punctuation + '\n\xa0«»\t—…'

# Removing the punctuation characters from the text
text = "".join([ch for ch in text if ch not in spec_chars])

# Defining a function to remove some specific characters from the text
def remove_chars_from_text(text, chars):
    return "".join([ch for ch in text if ch not in chars])

text = remove_chars_from_text(text, spec_chars)
text = remove_chars_from_text(text, string.digits)

# text tokenization
text_tokens = word_tokenize(text)
fText = nltk.Text(text_tokens)
fText = fText.tokens

# Defining the stop words we want to remove
russian_stopwords = stopwords.words("russian")
russian_stopwords.extend(['это', 'и', 'в', 'на', '↑', '↓'])

# Defining a function to delet Russian stop words
def remove_words_from_list(word_list, words_to_remove):
    # Check if 'word_list' is not None
    if word_list is not None:
        # Ensure all words are strings
        word_list = [str(word) for word in word_list]
        
        # Remove specified words
        filtered_words = [word for word in word_list if word.lower() not in words_to_remove]

        return filtered_words
    else:
        # Handle the case when 'word_list' is None
        return None

# Removing Russian stop words from the text    
result_text = remove_words_from_list(fText, russian_stopwords)
fdist = FreqDist(result_text)

# Printing the 10 most frequented words in the text
print("The 10 most frequented words are: ", '\n', fdist.most_common(10), '\n')

# Plotting the statistics of the occurrence of words in the text
fdist.plot(30,cumulative=False)

## Визуализация популярности слов в виде облака

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(fdist)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#####
# Initialize Natasha components
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)


# Process the text
doc = Doc(textF)
doc.segment(segmenter)
doc.tag_morph(morph_tagger)
doc.parse_syntax(syntax_parser)


# Function to extract text content from a URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    
    # Getting the names of articles we are searching about and storing them in a string array called "nameList"
    name_list = [h2.get_text(strip=True) for h2 in soup.findAll('h2', {'class': 'tm-title'})]

    # Getting the authors' names of these articles and storing them in a string array called "authors_list"
    authors_list = [a.get_text(strip=True) for a in soup.find_all('a', class_='tm-user-info__username')]

    # Zipping the name list and authors list into pairs
    pairs = list(zip(name_list, authors_list))

    return pairs

# the URL of our search in the electronic library web page
web_page_text = extract_text_from_url(myUrl)
print("The Key figures are: ", '\n')
print(web_page_text)

