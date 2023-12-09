import csv
import pickle
import numpy as np

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


def create_dictionary(messages):
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

    word_list = []

    # unique_words = list(set(messages))
    for message in messages:
        words = get_words(message)
        # print('words', words)

        word_list.extend(set(words))

        # print('word_list', word_list)
        # import pdb; pdb.set_trace()

    # print('word_list', word_list)
    unique_words_dict = dict([[x,word_list.count(x)] for x in set(word_list)])
    # unique_words_dict = {i:word_list.count(i) for i in word_list}
    for word in unique_words_dict:
        if unique_words_dict[word] >= 5: #count of messages[ii] in messages >= 5
            text_dict.update({word:idx}) # add word to dictionary
            idx += 1
        
    # print('dictionary first 10', (list(text_dict.keys())[:10]))
    # print('text_dict', text_dict)
    return text_dict
    # *** END CODE HERE ***


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

###############read csv###############
file = open('Scraped_data_news_output.csv', encoding='utf-8')
data = csv.reader(file)

data_list = []
for row in data:
    # import pdb; pdb.set_trace()
    # inter = get_words(listToString(row))
    inter = listToString(row)
    data_list.append(inter)

#################create word dictionary###############

textdic = create_dictionary(data_list)
# import pdb; pdb.set_trace()

with open('US_words_dictionary.pkl', 'wb') as handle:
    pickle.dump(textdic, handle, protocol=pickle.HIGHEST_PROTOCOL)

###############create data matrix#######################
datMat = transform_text(data_list, textdic)

with open('US_words_train_mat.pkl', 'wb') as handle:
    pickle.dump(datMat, handle, protocol=pickle.HIGHEST_PROTOCOL)


#################save to pickle file###############
# with open('US_words_dictionary.pickle', 'wb') as handle:
#     pickle.dump(textdic, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('US_words_train_mat.pickle', 'rb') as handle:
#     b = pickle.load(handle)


# import pdb; pdb.set_trace()
