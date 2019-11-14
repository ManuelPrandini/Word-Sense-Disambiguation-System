from tqdm import tqdm
from tensorflow.python.keras.utils import to_categorical
import numpy as np
import global_paths as gp
import parsing as par


def clean_data(data):
    '''
    Method used to clean the data (array of arrays of words)
    from the punctuation and other mistakes.
    I tried to do different cleaning on data. First, only removing the punctuation, then removing only
    some punctuation characters and the removing also some print mistakes at the end of the words( the part
    is commented)
    :param data the input array of arrays to clean
    :return an array of arrays with word cleaned
    '''
    result = []
    for sentence in data:
        row = []
        for word in sentence:
            if word not in par.chars_to_remove and '``' != word and "''" != word:
                row.append(word)
            '''
            if '``' == word or "''" == word or word in par.chars_to_remove:
                continue
            else:
          
                #many words end with various characters
                if word.endswith('.') or word.endswith('_') or word.endswith(','):
                    while len(word) > 0 and word[-1] in par.chars_to_remove:
                        word = word[:-1]
                    if word in par.chars_to_remove:
                        continue
                '''
        result.append(row)
    return result


def create_input_vocab(data,times):
    '''
    Method used to create the input vocab, a dictionary that contains for each word
    of the data passed in input the specific id token.
    :param data the array of arrays containing words taken from input data
    :param times variable used to take only words that are present inside the data array
    more than the value indicated
    :return the dictionary containing the mapping from words to integers.
    '''
    vocab = {'<PAD>': 0, '<UNK>': 1}
    freq = {}

    #calculate the number of appears of each word
    for sentence in data:
        for word in sentence:
            if word not in freq:
                freq[word] = 0
            freq[word]+=1

    filter_words = []
    #drop the words that appear less then times
    for k,v in freq.items():
        if v >= times :
            filter_words.append(k)

    #fill the vocab
    for word in filter_words:
        vocab[word] = len(vocab)

    return vocab


def create_sense_bn_vocab(data, times):
    '''
    Method used to create the sense vocab for the babelnet synsets, a dictionary that contains for each babelnet synset
    of the data passed in input the specific id token.
    :param data the array of arrays containing words taken from labels
    :param times variable used to take only teh babelnet synsets that are present inside the data array
    more than the value indicated
    :return the dictionary containing the mapping from babelnet synsets to integers.
    '''
    vocab = {}
    freq = {}

    for sentence in data:
        for word in sentence:
            if "_bn:" in word:
                if word not in freq:
                    freq[word] = 0
                freq[word]+=1

    filter_words = []
    # drop the words that appear less then times
    for k, v in freq.items():
        if v >= times:
            filter_words.append(k)

    # fill the vocab
    for word in filter_words:
        vocab[word] = len(vocab)

    return vocab


def create_sense_inventory(map):
    '''
    Method used to create the sense inventory map.
    This method is used both to create dom inventory and lexnames inventory.
    From the map passed in input takes the values.
    :param map the map that can be bn2lexnames or bn2domains
    :return the map that contains all the sense of the map passed in input
    with the corresponding id token
    '''
    vocab = {}
    for k,v in map.items():
        if v not in vocab:
            vocab[v] = len(vocab)
    return vocab


def make_output_bn_vocab(vocab_input_data, vocab_sense_labels):
    '''
    Method that creates the output_vocab and its reverse. It consists of the join of
    the input vocab and the sense bn vocab. Is used to tokenize the various
    datasets.
    :param vocab_input_data the dictionary containing only words without sense
    :param vocab_sense_labels the dictionary containing only sense words taken from the data labels.
    :return 2 outputs:
        - output_vocab that has as keys both normal words and sense words and as values the corresponding id tokens
        - reverse_output_vocab that has as key the values of output_vocab and as values the keys of output_vocab
    '''
    reverse_output_vocab = dict()
    output_vocab = dict()
    for k in vocab_input_data.keys():
        output_vocab[k] = len(output_vocab)
        reverse_output_vocab[len(reverse_output_vocab)] = k
    for k in vocab_sense_labels.keys():
        output_vocab[k] = len(output_vocab)
        reverse_output_vocab[len(reverse_output_vocab)] = k
    return output_vocab, reverse_output_vocab


def make_output_sense_vocab(vocab_sense_labels):
    '''
    Method that creates the output sense vocab and its reverse vocab adding the PAD e UNK entries.
    :param vocab_sense_labels the dictionary that can contains dom labels or lexnames labels and it is used
    to make the output vocab.
    :return 2 outputs:
            - vocab that has as keys the sense words  and as values the corresponding id tokens
            - reverse_vocab that has as key the id tokens and as values the  corresponding keys of vocab
    '''
    vocab = {'<PAD>':0,'<UNK>':1}
    reverse_vocab = {0:'<PAD>', 1:'<UNK>'}
    for k in vocab_sense_labels.keys():
        vocab[k] = len(vocab)
        reverse_vocab[len(reverse_vocab)] = k
    return vocab, reverse_vocab



def tokenize_data(data, output_vocab,mode="train"):
    '''
    Method that take in input the test_data as array of arrays words
    and produce an array of arrays of words converted in tokens through the
    vocab given in input.
    :param data an array of arrays of words
    :param output_vocab the dictionary containing the mapping from words to token ids
    :param mode the modality of how tokenize the input data. If the modality is 'train'
    is tokenized the normal word inside the array of the sentence.
    If the modality is 'test' , the word is splitted because the element of array
    contains more informations.
    :return a numpy array of numpy arrays of tokens (int)
    '''
    result = []
    for sentence in data:
        row = []
        for word in sentence:
            if mode == "train":
                if word in output_vocab:
                    row.append(int(output_vocab[word]))
                else:
                    row.append(int(output_vocab['<UNK>']))
            #the test data contains for each word the following format :
            #word|id_term|position|lemma|POS
            elif mode == "test":
                splitted = word.split("|")
                if splitted[0] in output_vocab:
                    row.append(int(output_vocab[splitted[0]]))
                else:
                    row.append(int(output_vocab['<UNK>']))
            else:
                raise Exception("invalid mode!")
        result.append(np.array(row))
    return np.array(result)


def load_data_from_file(file_to_read):
    '''
    Method used to load the data from a specific file and organizes the data
    as array of arrays.
    :param file_to_read the file.txt where read the data
    :return an array of arrays containing single words of each sentence
    '''
    result = []
    with open(file_to_read, 'r', encoding='utf-8') as fr:
        for sentence in tqdm(fr):
            row = []
            for word in sentence.strip().split(" "):
                row.append(word)
            result.append(row)
    fr.close()
    return result


def convert_to_categorical(labels,vocab):
    '''
    Method used to create the categorical shape of each label
    from the labels vector. Each label become a np.array
    :param labels the vector of labels
    :param vocab the dictionary of word index used to take the length. It needs for the number of classes of
    the categorical shape.
    :return a np.array of np.arrays containing the categorical shape conversion of each label
    '''
    result = []
    for label in labels:
        categorical_label = to_categorical(label,num_classes=len(vocab))
        result.append(np.array(categorical_label))
    return np.array(result)


def find_max_length(input_data):
    '''
    Method that finds the value of the sentence with the most words ( sentence with max length )  in the input_data.
    :param input_data array of arrays containing the words converted in id numbers and tokenized
    :return the value of the sentence with the most words as integer.
    '''
    max_length = 0
    i = 0
    for array in input_data:
        if(len(array)>max_length):
            max_length = len(array)
    return max_length



def find_max_length_sentence(sentence_array):
    '''
    Method that finds the sentence with max length in the sentence_array passed in input.
    Used to retrieve the value to pass in the method for pad the sentences when is used
    Elmo embeddings in the model
    :param sentence_array an array of sentences(strings)
    :return the max length sentence as integer
    '''
    max_length = 0
    for sentence in sentence_array:
        splitted = sentence[0].split(" ")
        if len(splitted) > max_length:
            max_length = len(splitted)
    return max_length


def add_pads_array(input_data,max_length):
    '''
    Method that add for each input_array of the input_data a pad_array of zero values until the max_length minus the length
    of the input_array
    :param input_data array of arrays containing the words converted in id numbers and tokenized
    :param max_length the  max length between all the input_arrays inside the input_data
    :return the input_data array containing the padded_arrays
    '''
    result = []
    for input_array in input_data:
        length = len(input_array)
        padded_array = np.pad(input_array,(0,max_length-length),mode='constant')
        result.append(padded_array)
    return result


def pad_sentence_array(sentence_data,max_length):
    '''
    Method used to pad the sentences to pass to Elmo Embeddings model.
    :param sentence_data array of sentences(strings)
    :param max_length the max length of the sentences in the sentence_data array
    return a numpy array of arrays where each array contains the sentence
    '''
    result = []
    for sentence in sentence_data:
        splitted = sentence[0].strip().split(" ")
        if len(splitted) >= max_length:
            result.append([" ".join(splitted[:max_length])])
        else:
            for i in range(0,max_length-len(splitted)):
                splitted.append('<PAD>')
            result.append([" ".join(splitted)])
    return np.array(result)


def make_sentences(data):
    '''
    Method used to make sentences from arrays of words. Each word
    is joined with the next word and they are separated by a blank space.
    :param data the array of arrays of words
    :return a numpy array of arrays where each array contains the sentence
    '''
    result = []
    for sentence in data:
        result.append([" ".join(sentence)])
    return np.array(result)



def make_sentences_for_test(data):
    '''
    Method used to make sentences from arrays of words. This method is similar to 'make_sentences()' but
     the words taken from the test (the instances), contain different informations like lemma, word, id position etc separated
     by "|" character.
     Each word is joined with the next word and they are separated by a blank space.
    :param data the array of arrays of words
    :return a numpy array of arrays where each array contains the sentence
    '''
    result = []
    for sentence in data:
        x = ""
        for word in sentence:
            word = word.split("|")
            x+=word[0] + " "
        result.append([x.strip()])
    return np.array(result)








