#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

os.chdir(r'C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\New folder')


# # reading the excel file 

# In[2]:


import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup


# Read the input.xlsx file
df_input = pd.read_excel(r"C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\Input.xlsx")


# # reading the stop words form the given text file
# 

# In[3]:


#reading stop words lists and putting them in different list

stop_words=[]

#reading and appending the file stopwords_auditor

with open(r"C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\StopWords-20230427T055245Z-001\StopWords\StopWords_Auditor.txt", 'r') as file:
    for line in file:
        stop_words.append(line.strip())

#reading and appending the file StopWords_Currencies

with open(r"C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\StopWords-20230427T055245Z-001\StopWords\StopWords_Currencies.txt", 'r') as file:
    for line in file:
        stop_words.append(line.strip())

#reading and appending the file StopWords_DatesandNumbers

with open(r"C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\StopWords-20230427T055245Z-001\StopWords\StopWords_DatesandNumbers.txt", 'r') as file:
    for line in file:
        stop_words.append(line.strip())

#reading and appending the file StopWords_Generic

with open(r"C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\StopWords-20230427T055245Z-001\StopWords\StopWords_Generic.txt", 'r') as file:
    for line in file:
        stop_words.append(line.strip())

#reading and appending the file StopWords_GenericLong

with open(r"C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\StopWords-20230427T055245Z-001\StopWords\StopWords_GenericLong.txt", 'r') as file:
    for line in file:
        stop_words.append(line.strip())

#reading and appending the file StopWords_Geographic

with open(r"C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\StopWords-20230427T055245Z-001\StopWords\StopWords_Geographic.txt", 'r') as file:
    for line in file:
        stop_words.append(line.strip())     

#reading and appending the file StopWords_Names

with open(r"C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\StopWords-20230427T055245Z-001\StopWords\StopWords_Names.txt", 'r') as file:
    for line in file:
        stop_words.append(line.strip())        


# # Creating a new directory to store the extracted data and Cleaning using Stop Words Lists

# In[4]:


# Create the directory to store extracted data
folder_path = r"C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\New folder\Extracted_Articles"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Create a directory to cleaned extracted data
cleaned_folder_path=r"C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\New folder\Extracted_Articles\cleaned"
    
if not os.path.exists(cleaned_folder_path):
    os.makedirs(cleaned_folder_path)


# # Extracting the data from the url and Cleaning using Stop Words Lists 

# In[5]:


#we will extract data from URLs and clean using stop words
#we will srote the extracted and clean extracted data in seperate directories


# Loop through each URL and extract the article text
for index, row in df_input.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    response = requests.get(url)
    if response.status_code == 404:
        print(f"Skipping URL_ID {url_id} because the page was not found")
        continue
    soup = BeautifulSoup(response.content, 'html.parser')
    article_title = soup.find('h1').get_text().strip()
    article_text = ''
    
    # Find the main article content
    main_content = soup.find('article')
    if main_content is not None:
        for p in main_content.find_all('p'):
            # Skip paragraphs containing contact or copyright information
            if 'contact' in p.get_text().lower() or 'copyright' in p.get_text():
                continue
            article_text += p.get_text().strip() + '\n'
    else:
        print(f"Skipping URL_ID {url_id} because the main article content could not be found")
        continue
    
    # Save the extracted article in a text file in the specified folder
    file_path = os.path.join(folder_path, f'{url_id}.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f'{article_title}\n\n{article_text}')
   
    
    # clean the data while     
    cleaned_title = ' '.join([word for word in article_title.split() if word.lower() not in stop_words])
    cleaned_text = ' '.join([word for word in article_text.split() if word.lower() not in stop_words])

    #save the cleaned data in a text file in the specified folder
    cleaned_file_path = os.path.join(cleaned_folder_path, f'cleaned_{url_id}.txt')
    with open(cleaned_file_path, 'w',encoding='utf-8') as file:
        file.write(f'{cleaned_title}\n\n{cleaned_text}')    


# # creating positive and negative words dictionary

# In[6]:


#reading positive and negative words text file and putting them in different set 
#we are creating a dictionary for positive and negative words
#We add only those words in the dictionary 
#if they are not found in the Stop Words Lists.


positive_words = set()
negative_words = set()

with open(r"C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\MasterDictionary-20230427T055244Z-001\MasterDictionary\positive-words.txt", 'r') as f:
    for word in f.read().split():
        if word not in stop_words:
            positive_words.add(word.lower())

with open(r"C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\MasterDictionary-20230427T055244Z-001\MasterDictionary\negative-words.txt", 'r') as f:
    for word in f.read().split():
        if word not in stop_words:
            negative_words.add(word.lower())


# In[7]:


#creating a dictionary

dictionary = {'positive': positive_words, 'negative': negative_words}

import csv

# Writing dictionary of positive and negative words to csv file

with open(r'C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\New folder\dictionary_positive_negative_words.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Positive Words'] + list(positive_words))
    writer.writerow(['Negative Words'] + list(negative_words))


# # reading the given output data structure file

# In[69]:


#reading the excel file  output data structure
df_output = pd.read_excel(r"C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\Output Data Structure.xlsx")


# # Extracting Derived variables 

# CALCULATING POSITIVE SCORE , NEGATIVE SCORE , POLARITY SCORE , SUBJECTIVITY SCORE
# 

# In[70]:


#creating an empty list to put the values we get after calculating
#loop through the files and appending the values we get to the list


# In[71]:


#creating empty list to add values
positive_score = []
negative_score = []
polarity_score = []
subjectivity_score = []


# In[72]:


import nltk
from nltk.tokenize import word_tokenize


# Loop over all cleaned text files
# since we have named the files 
for i in range(37, 151):
    # Open the cleaned file
    cleaned_file_path = f'C:/Users/ishan/OneDrive/Desktop/internshala/blackcoffer_internshala/New folder/Extracted_Articles/cleaned/cleaned_{i}.txt'
    try:
        with open(cleaned_file_path, 'r', encoding='utf-8') as file:
            cleaned_text = file.read()
            # process the file contents here
            # Tokenize the text
            tokens = word_tokenize(cleaned_text.lower())
    
            # Calculate the scores
            pos_score = sum(1 for word in tokens if word in dictionary["positive"] )
            positive_score.append(pos_score)
    
            neg_score = sum(1 for word in tokens if word in dictionary["negative"] ) * -1
            negative_score.append(neg_score)
    
            polarity_sco_re = (pos_score - neg_score) / ((pos_score + neg_score) + 0.000001)
            polarity_score.append(polarity_sco_re)
    
            subjectivity_sco_re = (pos_score + abs(neg_score)) / (len(tokens) + 0.000001)
            subjectivity_score.append(subjectivity_sco_re)
    
    except FileNotFoundError:
        positive_score.append(np.nan)
        negative_score.append(np.nan)
        polarity_score.append(np.nan)
        subjectivity_score.append(np.nan)
        
   


# In[73]:


#adding the values of the list to the dataframe names df_output


# In[74]:


df_output["POSITIVE SCORE"]=positive_score
df_output["NEGATIVE SCORE"]=negative_score
df_output["POLARITY SCORE"]=polarity_score
df_output["SUBJECTIVITY SCORE"]=subjectivity_score


# # 	Analysis of Readability

# Average Number of Words Per Sentence

# ### CLACULATING    	AVG SENTENCE LENGTH , PERCENTAGE OF COMPLEX WORDS , FOG INDEX , AVG NUMBER OF WORDS PER SENTENCE , COMPLEX WORD COUNT

# In[75]:


#creating an empty list to put the values we get after calculating
#loop through the files and appending the values we get to the list


# In[76]:


avg_sentence_length=[]
pct_complex_words=[]
fog_index=[]
avg_words_per_sentence=[]
num_complex_words=[]


# In[77]:


import re

# loop through all cleaned files in the directory
for i in range(37, 151):
    # read in the text file
    cleaned_file_path_1 = f'C:/Users/ishan/OneDrive/Desktop/internshala/blackcoffer_internshala/New folder/Extracted_Articles/cleaned/cleaned_{i}.txt'
    try :
        
        with open(cleaned_file_path_1, 'r',encoding='utf-8') as file:
            text = file.read()

        # calculate the number of sentences
        sentences = re.split(r'[.!?]+', text)
        num_sentences = len(sentences)

        # calculate the number of words
        words = text.split()
        num_words = len(words)

        # define a function to determine if a word is complex
        def is_complex(word):
            num_syllables = len(re.findall(r'[aeiouy]+', word, re.I))
            return num_syllables > 2

        # calculate the number of complex words
        num_complex_wor_ds = len([word for word in words if is_complex(word)])
        num_complex_words.append(num_complex_wor_ds)
        
        
        # calculate the average sentence length
        avg_sentence_len_gth = num_words / num_sentences
        avg_sentence_length.append(avg_sentence_len_gth)

        
        # calculate the percentage of complex words
        pct_complex_wor_ds = num_complex_wor_ds / num_words
        pct_complex_words.append(pct_complex_wor_ds)

        
        # calculate the Fog Index
        fog_in_dex = 0.4 * (avg_sentence_len_gth + pct_complex_wor_ds)
        fog_index.append(fog_in_dex)

        
        # calculate the average number of words per sentence
        avg_words_per_sen_tence = num_words / num_sentences
        avg_words_per_sentence.append(avg_words_per_sen_tence)
        
    except FileNotFoundError:
        avg_sentence_length.append(np.nan)
        pct_complex_words.append(np.nan)
        fog_index.append(np.nan)
        avg_words_per_sentence.append(np.nan)
        num_complex_words.append(np.nan)


# In[78]:


#adding the values of the list to the dataframe names df_output


# In[79]:


df_output["AVG SENTENCE LENGTH"]=avg_sentence_length
df_output["PERCENTAGE OF COMPLEX WORDS"]=negative_score
df_output["FOG INDEX"]=fog_index

df_output["AVG NUMBER OF WORDS PER SENTENCE"]=avg_words_per_sentence
df_output["COMPLEX WORD COUNT"]=num_complex_words


# # 	Word Count

# claculating total number of words

# In[80]:


#creating an empty list to put the values we get after calculating
#loop through the files and appending the values we get to the list


# In[81]:


import os
import nltk
from nltk.corpus import stopwords
import string


# In[82]:


word_counts = []


# In[83]:


def count_words_in_files():
    # set the path to the directory containing the files
    # initialize a list to store the word counts for each file
    # loop through all cleaned files in the directory
    for i in range(37, 151):
        cleaned_file_path_2 = f'C:/Users/ishan/OneDrive/Desktop/internshala/blackcoffer_internshala/New folder/Extracted_Articles/cleaned/cleaned_{i}.txt'
        try :    
            with open(cleaned_file_path_2, 'r',encoding='utf-8') as f:
                text = f.read()
            # convert text to lowercase
            text = text.lower()

            # remove punctuation marks
            text = text.translate(str.maketrans('', '', string.punctuation))

            # tokenize the text into words
            words = nltk.word_tokenize(text)

            # remove stopwords
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words]

            # count the remaining words
            num_words = len(words)

            # add the word count to the list
            word_counts.append(num_words)

        except FileNotFoundError:
            word_counts.append(np.nan)
            
count_words_in_files()


# In[84]:


df_output["WORD COUNT"]=word_counts


# In[85]:


df_output.head(15)


# CALCULATING THE VALUES OF SYLLABLE PER WORD , PERSONAL PRONOUNS , AVG WORD LENGTH
# 

# In[ ]:





# In[86]:


# initialize lists to store the metrics for each file

syllable_counts = []
personal_pronouns_counts = []
avg_word_lengths = []


# In[91]:


def analyze_files():    

    # loop through all cleaned files in the directory

    for i in range(37, 151):
        
        # set the path to the directory containing the files
        cleaned_file_path_3 = f'C:/Users/ishan/OneDrive/Desktop/internshala/blackcoffer_internshala/New folder/Extracted_Articles/cleaned/cleaned_{i}.txt'
        try :         
            with open(cleaned_file_path_3, 'r',encoding='utf-8') as f:
                text = f.read()
            # convert text to lowercase
            text = text.lower()

            # count syllables in each word
            words = text.split()
            num_syllables = 0

            for word in words:                
                # handle exceptions for words ending with "es" or "ed"
                if word[-2:] in ['es', 'ed']:
                    word = word[:-2]

                # count vowels in word
                vowels = 'aeiouy'
                num_vowels = 0
                prev_char = ''

                for char in word:
                    if char in vowels and prev_char not in vowels:
                        num_vowels += 1
                    prev_char = char

                if word.endswith('e'):
                    num_vowels -= 1

                if num_vowels == 0:
                    num_vowels = 1
                num_syllables += num_vowels
            syllable_counts.append(num_syllables)

            # count personal pronouns
            personal_pronouns = re.findall(r'\b(I|we|my|ours|us)\b', text)
            personal_pronouns_counts.append(len(personal_pronouns))

            # calculate average word length
            words = text.split()
            total_word_length = sum(len(word) for word in words)
            avg_word_length = total_word_length / len(words)
            avg_word_lengths.append(avg_word_length)

        except FileNotFoundError:
            syllable_counts.append(np.nan)
            personal_pronouns_counts.append(np.nan)
            avg_word_lengths.append(np.nan)
            
            
analyze_files()


# In[95]:


df_output["SYLLABLE PER WORD"]=syllable_counts
df_output["PERSONAL PRONOUNS"]=personal_pronouns_counts
df_output["AVG WORD LENGTH"]=avg_word_lengths


# In[96]:


df_output.head(15)


# In[98]:


df_output.fillna("page not found", inplace=True)


# In[100]:


df_output.head(15)


# In[104]:


output_folder_path = r"C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\New folder\output"

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

df_output.to_csv(r"C:\Users\ishan\OneDrive\Desktop\internshala\blackcoffer_internshala\New folder\output\output_data_structure.csv",index=False)


# In[ ]:




