import csv
import pickle
import numpy as np

from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    # print('pre-norm', message[0:10])
    message.lower().split()
    # print('post-norm', message[0:10])

    # import pdb; pdb.set_trace()
    return message.lower().split()
    # print('message', message)

    # *** END CODE HERE ***


def create_dictionary(messages, stop_word_list, stop_symbol_list):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***

    text_dict = {}
    idx = 0
    count = 0

    word_list = []

    for message in messages:
        words = get_words(message)
        # print('words', words)

        # import pdb; pdb.set_trace()

        #### pre-process messages 

        ################## remove words in stoplist###################
        new_words = [word for word in words if word not in stop_word_list]

        ################## remove words in stop symbol##############
        new_words = remove_words(new_words, stop_symbol_list)

        ################### mainly for tweets, remove nonascii words####################

        # import pdb; pdb.set_trace()

        new_words = is_ascii(new_words)
        # import pdb; pdb.set_trace()

        ##################lemmatize words########################

        lemmatizer = WordNetLemmatizer()

        # import pdb; pdb.set_trace()

        new_words = [lemmatizer.lemmatize(word) for word in new_words]

        # import pdb; pdb.set_trace()



        ################# remove words that are just spaces ######################
        # import pdb; pdb.set_trace()
        new_words = [ele for ele in new_words if ele.strip()]
        # import pdb; pdb.set_trace()
        

        word_list.extend(set(new_words))

        # import pdb; pdb.set_trace()

        # print('word_list', word_list)
        # import pdb; pdb.set_trace()

        count += 1
        if count % 100 == 0:
            print("message number =", count)

    count1 = 0

    # print("finished part 1")

    # print('word_list', word_list)


   #################### throw out words with count < 5 ########################
    # unique_words_dict = dict([[x, word_list.count(x)] for x in set(word_list)])
    unique_words_dict = Counter(word_list)
    # import pdb; pdb.set_trace()

    for word in unique_words_dict:
        if unique_words_dict[word] >= 5: #count of messages[ii] in messages >= 5
            text_dict.update({word:idx}) # add word to dictionary
            idx += 1

        count1 += 1
        if count1 % 100 == 0:
            print("create dict pt 2 message number =", count)
    ############################################################################
        
    # print('dictionary first 10', (list(text_dict.keys())[:10]))
    # print('text_dict', text_dict)
    return text_dict
    # *** END CODE HERE ***

def remove_words(in_list, bad_list):
    out_list = []
    for line in in_list:
        words = ' '.join([word for word in line.split() if not any([phrase in word for phrase in bad_list]) ])
        out_list.append(words)
    return out_list[1:]

def is_ascii(word_single_message):
    # import pdb; pdb.set_trace()
    normal_word = []
    for word in word_single_message:
        for char in word:
            if ord(char) >= 128:
                print("non-ascii detected")
                break

        # import pdb; pdb.set_trace()
        normal_word.append(word)

    # return all(ord(c) < 128 for c in s)

    # import pdb; pdb.set_trace()
    return normal_word

def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    # max(word_dictionary[:]
    # unique_words_dict = {i:messages.count(i) for i in messages}
    ii_max = len(messages)
    jj_max = len(word_dictionary)
    count = 0

    rows, cols = (ii_max, jj_max)
    word_array = np.zeros((rows, cols))

    # print('unique words dict', unique_words_dict)

    for ii in range(0, ii_max):
        single_message_list = get_words(messages[ii])
        # import pdb; pdb.set_trace()

        words_dict_per_message = {i:single_message_list.count(i) for i in single_message_list}        
    
        for word in single_message_list:
            if word in word_dictionary:
                jj = word_dictionary[word]
                word_array[ii, jj] = words_dict_per_message[word]

        count += 1
        if count % 100 == 0:
            print("transform text message number =", count)

    # print('sum+_word_array', np.sum(word_array))

    # print('word_array', word_array[0][0:100])
    
    return word_array
    # *** END CODE HERE ***


def listToString(s):
 
    # initialize an empty string
    str1 = ""
 
    # traverse in the string
    for ele in s:
        str1 += ele
 
    # return string
    return str1


############## main ###############
# csv file is a list of strings

############### get policies ###############
# file = open('scraped_data_news_output.csv', encoding='utf-8')
# data = csv.reader(file)

# data_list = []
# for row in data:
#     # import pdb; pdb.set_trace()
#     # inter = get_words(listToString(row))
#     inter = listToString(row)
#     data_list.append(inter)

# import pdb; pdb.set_trace()

################ get tweets #######################


data_list=[]         #an empty list to store the second column
with open('twitter_sentiment_data.csv', encoding='utf-8') as rf:
    reader = csv.reader(rf, delimiter=',')
    for row in reader:
      data_list.append(row[1])

# import pdb; pdb.set_trace()

## ############## get stop word list #######################
with open('stopwords.txt', 'r') as file:
    # Create an empty list to store the lines
    stop_word_list = []

    # Iterate over the lines of the file
    for line in file:
        # Remove the newline character at the end of the line
        line = line.strip()

        # Append the line to the list
        stop_word_list.append(line)

# remove the first word
data_list = data_list[1:]

# Print the list of lines
# print(lines)

################### get stop symbol list ###################
with open('stopsymbols.txt', 'r') as file:
    # Create an empty list to store the lines
    stop_symbol_list = []

    # Iterate over the lines of the file
    for line in file:
        # Remove the newline character at the end of the line
        line = line.strip()

        # Append the line to the list
        stop_symbol_list.append(line)


# import pdb; pdb.set_trace()
#################create word dictionary###############

textdic = create_dictionary(data_list, stop_word_list, stop_symbol_list)
# import pdb; pdb.set_trace()

with open('US_tweets_dictionary.pkl', 'wb') as handle:
    pickle.dump(textdic, handle, protocol=pickle.HIGHEST_PROTOCOL)

###############create data matrix#######################
datMat = transform_text(data_list, textdic)

with open('US_tweets_train_mat.pkl', 'wb') as handle:
    pickle.dump(datMat, handle, protocol=pickle.HIGHEST_PROTOCOL)


#################save to pickle file###############
# with open('US_words_dictionary.pickle', 'wb') as handle:
#     pickle.dump(textdic, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('US_words_train_mat.pickle', 'rb') as handle:
#     b = pickle.load(handle)


# import pdb; pdb.set_trace()
