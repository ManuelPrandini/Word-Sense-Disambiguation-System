import model as md
import preprocessing as pre
import os
import numpy as np
from tqdm import tqdm
from nltk.corpus import wordnet as wn
import parsing as par
import global_paths as gp
import tensorflow as tf
from tensorflow.python.keras import backend as K, metrics


#DEFINE BEST MODEL PARAMETERS
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 256
INPUT_DROPOUT = 0.5
LSTM_DROPOUT = 0.2

#DEFINE THE CONFIGURATIONS FOR THE MODEL USED
PREDICT_ELMO = True
PREDICT_ATTENTION = True
PREDICT_2BLSTM = False

weights_file = "weights.hdf5"


def return_index_of_vocab_bn(lemma, output_vocab, sense_vocab):
    '''
    Method that return only the corresponding tokens form the output_vocab from the synsets that have the same lemma passed
    in input.
    :param lemma the input word
    :param output_vocab the dictionary that contains simple words and sense words as keys
    and id tokens as values
    :param sense_vocab the dictionary that contains the babelnet synsets used as filter.
    :return a list with the id tokens
    '''
    result = []
    for k,v in output_vocab.items():
        #if the key is a sense word and the lemma of this word
        #is equal to my lemma
        if k in sense_vocab and k.split("_bn:")[0] == lemma:
            result.append(v)
    return result


def return_all_synsets_from_lemma(lemma,output_bn_vocab):
    '''
    Method used to return all synsets from the output_bn_vocab that have the same lemma passed in input.
    :param lemma lemma used to search the corresponding synsets
    :param output_bn_vocab dictionary where search the babelnet synsets
    :return a list containing only babelnet synsets that corresponding to the lemma
    '''
    result = []
    for k in output_bn_vocab.keys():
        if "bn:" in k and k.split("_bn:")[0] == lemma:
            result.append("bn:"+k.split("_bn:")[1])
    return result


def return_sense_indices_of_vocab(list_synsets, map_to_convert, output_sense_vocab, mode = 'dom'):
    '''
    Method that takes in input a list of babelnet synset and return a list of corresponding id tokens of
    wordnet domains or lexnames, depending by the mode indicated.
    :param list_synsets a list containing babelnet synsets
    :param map_to_convert map that contains the mapping from babelnet synsets to wndomains or lexnames
    :param output_sense_vocab the dictionary used to map the label file. It can contains lexnames or wndomains
    :param mode can has 2 values:
            - 'dom' : indicates that if an element inside the list_synsets is not present in map_to_convert
            then set the sense to reseach in the output_sense_vocab as 'factotum'
            - 'lex' : indicates that if an element inside the list_synsets is not present in map_to_convert
            then return an empty list
    :return a list of id tokens(indices) of the output_sense_vocab
    '''
    result = []
    for el in list_synsets:
        if el in map_to_convert:
            sense = map_to_convert[el]
        else:
            if mode == 'dom':
                sense = "factotum"
            elif mode == 'lex':
                return []
        index = output_sense_vocab[sense]
        result.append(index)
    return result



def write_result_on_file(file_to_write,result_map):
    '''
    Method that writes the result obtained from the predict on a file.
    :param file_to_write the file_path where write the result
    :param result_map the dictionary containing the id of the instances as keys and
    the value predicted as values
    '''
    with open(file_to_write,'w',encoding='utf-8') as fw:
        for k, v in result_map.items():
            fw.write(k+" "+v+'\n')
    fw.close()


def preprocess_for_predict(input_path, resources_path, mode = 'bn'):
    '''
    Common method used in all the predicts that preprocess the input xml file used for the test,
    and tokenizes the input if the model to test is build with Keras Embeddings, else prepares the input
    as normal strings. Then creates the right vocab  that are used to define the model and load the trained weights on it.
    At the end, based on the mode chosen, return the right output.
    :param input_path: the path of the xml file to test
    :param resources_path: the resource folder path of the project
    :param mode: there are 3 modes that decide what return as values from the method. They are: 'bn','dom','lex'
    :return: different values based on the mode chosen in input. But principally an array of arrays of words where each word
    has more informations like position, lemma etc; the text tokenized to pass in input at the model predict; the input vocab; the sense vocab; the corresponding
    output vocab based on the mode chosen; the reverse output vocab to retrieve the right sense from the id token; the map from 'wordnet-babelnet' synsets and
    the map from 'babelnet synset - sense' based on the mode chosen.
    '''
    print("load data and clean it to make vocabs")
    # load the paths used to make the vocabs
    #I used commented parse when I tested the concatenation of datasets and only TOM
    train_x = pre.load_data_from_file(os.path.join(resources_path,gp.semCor_train_input_path)) # pre.load_data_from_file(gp.resources_path + gp.tom_resized_input_path)
    label_bn_y = pre.load_data_from_file(os.path.join(resources_path, gp.semCor_train_bn_labels_path))#pre.load_data_from_file(gp.resources_path + gp.tom_resized_bn_labels_path)


    #load maps to create output domains vocab and lexnames vocab
    b2dom, dom2b = par.create_maps_from_tsv(os.path.join(resources_path, gp.b2wn_domains_map_path))
    b2lex, lex2b = par.create_maps_from_tsv(os.path.join(resources_path, gp.b2lexnames_map_path))

    print("create wordnet to babelnet mapping")
    # create the wordnet to babelnet mapping
    b2w_map, w2b_map = par.create_maps_from_tsv(os.path.join(resources_path, "babelnet2wordnet.tsv"))

    print("load input test")
    # load datasets for the test
    test_x = par.parse_test_data(input_path)

    print("create vocabs")
    number_of_appearance_input = 5
    number_of_appearance_sense_bn = 5

    vocab_input_data = pre.create_input_vocab(train_x,number_of_appearance_input)
    vocab_bn_labels = pre.create_sense_bn_vocab(label_bn_y, number_of_appearance_sense_bn)
    vocab_dom_labels = pre.create_sense_inventory(b2dom)
    vocab_lex_labels = pre.create_sense_inventory(b2lex)
    output_bn_vocab, reverse_output_bn_vocab = pre.make_output_bn_vocab(vocab_input_data, vocab_bn_labels)
    output_dom_vocab, reverse_output_dom_vocab = pre.make_output_sense_vocab(vocab_dom_labels)
    output_lex_vocab, reverse_output_lex_vocab = pre.make_output_sense_vocab(vocab_lex_labels)

    #if not use elmo, tokenize data producing array of arrays of id tokens, else
    #produce array where each array inside contain the sentence as string separated by white space.
    if not PREDICT_ELMO:
        print("tokenize test data")
        # produce the test_data_tokenized
        test_x_tokenized = pre.tokenize_data(test_x, output_bn_vocab, mode='test')
    else:
        test_x_tokenized = pre.make_sentences_for_test(test_x)

    #return different variables base on the modality chosen
    if mode == 'bn':
        return test_x, test_x_tokenized, vocab_input_data, vocab_bn_labels, \
               output_bn_vocab,output_dom_vocab,output_lex_vocab, reverse_output_bn_vocab, w2b_map,
    elif mode == 'dom':
        return test_x, test_x_tokenized,vocab_input_data, vocab_dom_labels,\
               output_bn_vocab,output_dom_vocab,output_lex_vocab, reverse_output_dom_vocab, w2b_map, b2dom
    elif mode == 'lex':
        return test_x, test_x_tokenized,vocab_input_data, vocab_lex_labels,\
               output_bn_vocab,output_dom_vocab,output_lex_vocab, reverse_output_lex_vocab, w2b_map, b2lex


def predict_babelnet(input_path: str, output_path: str, resources_path: str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of synsets indices the resources folder containing your model and stuff you might need.
    :return: None
    """
    print("PREDICT BABELNET!")
    test_x, test_x_tokenized,vocab_input_data, vocab_bn_labels, output_bn_vocab,output_dom_vocab,output_lex_vocab, reverse_output_bn_vocab, w2b_map = \
        preprocess_for_predict(input_path,resources_path,mode='bn')

    print("build model")
    # build the model
    if PREDICT_2BLSTM:
        model = md.create_2_BLSTM_layers(len(vocab_input_data), len(output_bn_vocab), len(output_dom_vocab),
                                                      len(output_lex_vocab),
                                                      EMBEDDING_SIZE,
                                                      HIDDEN_SIZE,
                                                      INPUT_DROPOUT,
                                                      LSTM_DROPOUT)
    else:
        model = md.create_BLSTM(len(vocab_input_data), len(output_bn_vocab), len(output_dom_vocab),
                                                      len(output_lex_vocab),
                                                      EMBEDDING_SIZE,
                                                      HIDDEN_SIZE,
                                                      INPUT_DROPOUT,
                                                      LSTM_DROPOUT,elmo=PREDICT_ELMO,attention=PREDICT_ATTENTION)

    sess = K.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)
    print("load weights")
    # load the weights
    model.load_weights(os.path.join(resources_path, weights_file))

    print("predict the test")
    # predict from the test
    result = dict()
    num_words = 0
    num_mfs = 0
    # for each sentence
    for i in tqdm(range(len(test_x_tokenized))):
        # produce the probability matrix of predict
        pred_matrix = model.predict(test_x_tokenized[i])


        # take the indices of the instances of the sentence
        for word in test_x[i]:
            splitted = word.split("|")
            # only instances have splitted length major of 1
            if len(splitted) > 1:
                w = splitted[0]
                id = splitted[1]
                index = splitted[2]
                lemma = splitted[3]
                pos = splitted[4]

                num_words += 1

                if pos == 'NOUN':
                    pos = 'n'
                elif pos == 'VERB':
                    pos = 'v'
                elif pos == 'ADJ':
                    pos = 'a'
                elif pos == 'ADV':
                    pos = 'r'

                list_indices_vocab = return_index_of_vocab_bn(lemma, output_bn_vocab, vocab_bn_labels)

                # if there aren't synsets, in the mode case 'bn', return the mfs of the lemma
                if len(list_indices_vocab) == 0:
                    synset = par.mfs(lemma,pos, w2b_map)
                    result[id] = synset
                    num_mfs += 1
                else:
                    list_argmax = []
                    # take the values that interest for the argmax value
                    for val in list_indices_vocab:
                        if not PREDICT_ELMO:
                            list_argmax.append(pred_matrix[0][int(index)][0][val])
                        else:
                            list_argmax.append(pred_matrix[0][0][int(index)][val])
                    # compute argmax for specific word
                    argmax = np.argmax(list_argmax)
                    # obtain the correspettive synset
                    synset = reverse_output_bn_vocab[list_indices_vocab[argmax]]
                    # cut the lemma and take0 only the sense
                    sense = "bn:" + synset.split("_bn:")[1]
                    result[id] = sense
    print("mfs: ", num_mfs, " / ", num_words)

    print("write result on file")
    # HERE WRITE THE RESULT ON THE OUTPUT PATH
    write_result_on_file(os.path.join(output_path), result)
    print("create file in:", os.path.join( output_path))
    print("Done!")


def predict_wordnet_domains(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    print("PREDICT WORDNET DOMAINS!")
    test_x, test_x_tokenized, vocab_input_data, vocab_dom_labels, output_bn_vocab, output_dom_vocab, output_lex_vocab, reverse_output_dom_vocab, w2b_map, b2dom = \
        preprocess_for_predict(input_path, resources_path, mode='dom')

    print("build model")
    # build the model
    if PREDICT_2BLSTM:
        model = md.create_2_BLSTM_layers(len(vocab_input_data), len(output_bn_vocab), len(output_dom_vocab),
                                         len(output_lex_vocab),
                                         EMBEDDING_SIZE,
                                         HIDDEN_SIZE,
                                         INPUT_DROPOUT,
                                         LSTM_DROPOUT)
    else:
        model = md.create_BLSTM(len(vocab_input_data), len(output_bn_vocab), len(output_dom_vocab),
                                len(output_lex_vocab),
                                EMBEDDING_SIZE,
                                HIDDEN_SIZE,
                                INPUT_DROPOUT,
                                LSTM_DROPOUT, elmo=PREDICT_ELMO, attention=PREDICT_ATTENTION)

    sess = K.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)
    print("load weights")
    # load the weights
    model.load_weights(os.path.join(resources_path, weights_file))

    print("predict the test")
    # predict from the test
    result = dict()
    num_words = 0
    num_mfs = 0

    # for each sentence
    for i in tqdm(range(len(test_x_tokenized))):

        # produce the probability matrix of predict
        pred_matrix = model.predict(test_x_tokenized[i])

        # take the indices of the instances of the sentence
        for word in test_x[i]:
            splitted = word.split("|")
            # only instances have splitted length major of 1
            if len(splitted) > 1:
                w = splitted[0]
                id = splitted[1]
                index = splitted[2]
                lemma = splitted[3]
                pos = splitted[4]

                num_words += 1

                if pos == 'NOUN':
                    pos = 'n'
                elif pos == 'VERB':
                    pos = 'v'
                elif pos == 'ADJ':
                    pos = 'a'
                elif pos == 'ADV':
                    pos = 'r'

                list_synsets = return_all_synsets_from_lemma(lemma, output_bn_vocab)
                if len(list_synsets) == 0:
                    synset = par.mfs(lemma,pos, w2b_map)
                    if synset in b2dom:
                        result[id] = b2dom[synset]
                        num_mfs+=1
                    else:
                        result[id] = "factotum"
                    continue

                list_indices_vocab = return_sense_indices_of_vocab(list_synsets,b2dom,output_dom_vocab,mode='dom')
                if len(list_indices_vocab) == 0:
                    synset = par.mfs(lemma,pos, w2b_map)
                    if synset in b2dom:
                        result[id] = b2dom[synset]
                        num_mfs += 1
                    else:
                        result[id] = "factotum"
                    continue
                else:
                    list_argmax = []
                    # take the values that interest for the argmax value
                    for val in list_indices_vocab:
                        if not PREDICT_ELMO:
                            list_argmax.append(pred_matrix[1][int(index)][0][val])
                        else:
                            list_argmax.append(pred_matrix[1][0][int(index)][val])
                    # compute argmax for specific word
                    argmax = np.argmax(list_argmax)
                    # obtain the correspettive synset
                    synset = reverse_output_dom_vocab[list_indices_vocab[argmax]]
                    result[id] = synset
    print("MFS SU TOT WORDS: ",num_mfs, " / ",num_words)

    print("write result on file")
    # HERE WRITE THE RESULT ON THE OUTPUT PATH
    write_result_on_file(os.path.join(output_path), result)
    print("create file in:", os.path.join( output_path))
    print("Done!")


def predict_lexicographer(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    print("PREDICT LEXNAMES!")
    test_x, test_x_tokenized, vocab_input_data, vocab_lex_labels, output_bn_vocab, output_dom_vocab, output_lex_vocab, reverse_output_lex_vocab, w2b_map, b2lex = \
        preprocess_for_predict(input_path, resources_path, mode='lex')

    print("build model")
    # build the model
    if PREDICT_2BLSTM:
        model = md.create_2_BLSTM_layers(len(vocab_input_data), len(output_bn_vocab), len(output_dom_vocab),
                                         len(output_lex_vocab),
                                         EMBEDDING_SIZE,
                                         HIDDEN_SIZE,
                                         INPUT_DROPOUT,
                                         LSTM_DROPOUT)
    else:
        model = md.create_BLSTM(len(vocab_input_data), len(output_bn_vocab), len(output_dom_vocab),
                                len(output_lex_vocab),
                                EMBEDDING_SIZE,
                                HIDDEN_SIZE,
                                INPUT_DROPOUT,
                                LSTM_DROPOUT, elmo=PREDICT_ELMO, attention=PREDICT_ATTENTION)

    sess = K.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)
    print("load weights")
    # load the weights
    model.load_weights(os.path.join(resources_path, weights_file))

    print("predict the test")
    # predict from the test
    result = dict()
    num_words = 0
    num_mfs = 0

    # for each sentence
    for i in tqdm(range(len(test_x_tokenized))):

        # produce the probability matrix of predict
        pred_matrix = model.predict(test_x_tokenized[i])

        # take the indices of the instances of the sentence
        for word in test_x[i]:
            splitted = word.split("|")
            # only instances have splitted length major of 1
            if len(splitted) > 1:
                w = splitted[0]
                id = splitted[1]
                index = splitted[2]
                lemma = splitted[3]
                pos = splitted[4]

                num_words += 1

                if pos == 'NOUN':
                    pos = 'n'
                elif pos == 'VERB':
                    pos = 'v'
                elif pos == 'ADJ':
                    pos = 'a'
                elif pos == 'ADV':
                    pos = 'r'

                list_synsets = return_all_synsets_from_lemma(lemma, output_bn_vocab)
                if len(list_synsets) == 0:
                    synset = par.mfs(lemma,pos, w2b_map)
                    if synset in b2lex:
                        result[id] = b2lex[synset]
                        num_mfs += 1
                    continue

                list_indices_vocab = return_sense_indices_of_vocab(list_synsets, b2lex, output_lex_vocab,mode='lex')
                if len(list_indices_vocab) == 0:
                    synset = par.mfs(lemma,pos, w2b_map)
                    if synset in b2lex:
                        result[id] = b2lex[synset]
                        num_mfs += 1
                    else:
                        result[id] = "adj.all"
                    continue
                list_argmax = []
                # take the values that interest for the argmax value
                for val in list_indices_vocab:
                    if not PREDICT_ELMO:
                        list_argmax.append(pred_matrix[2][int(index)][0][val])
                    else:
                        list_argmax.append(pred_matrix[2][0][int(index)][val])
                # compute argmax for specific word
                argmax = np.argmax(list_argmax)
                # obtain the correspettive synset
                synset = reverse_output_lex_vocab[list_indices_vocab[argmax]]
                result[id] = synset
    print("MFS SU TOT WORDS: ", num_mfs, " / ", num_words)

    print("write result on file")
    # HERE WRITE THE RESULT ON THE OUTPUT PATH
    write_result_on_file(os.path.join(output_path), result)
    print("create file in:", os.path.join(output_path))
    print("Done!")


def predict_MFS(input_path, output_path, resources_path, bn2sense_map, mode='bn'):
    '''
    Method used to predict all MFS for each instance of the input test file.
    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :param bn2sense_map: the dictionary that contains the babelnet synset to sense to predict mapping.
    :param mode: there are 3 modes that decide what return as values from the method. They are: 'bn','dom','lex'
    :return: None
    '''
    print("PREDICT MFS BN!")
    test_x, test_x_tokenized, vocab_input_data, vocab_bn_labels, output_bn_vocab, output_dom_vocab, output_lex_vocab, reverse_output_bn_vocab, w2b_map = \
        preprocess_for_predict(input_path, resources_path, mode=mode)

    print("predict the test")
    # predict from the test
    result = dict()
    num_words = 0
    num_mfs = 0

    # for each sentence
    for sentence in test_x:

        # take the indices of the instances of the sentence
        for word in sentence:
            splitted = word.split("|")
            # only instances have splitted length major of 1
            if len(splitted) > 1:
                w = splitted[0]
                id = splitted[1]
                index = splitted[2]
                lemma = splitted[3]
                pos = splitted[4]

                if pos == 'NOUN':
                    pos = 'n'
                elif pos == 'VERB':
                    pos = 'v'
                elif pos == 'ADJ':
                    pos = 'a'
                elif pos == 'ADV':
                    pos = 'r'

                num_words += 1

                synset = par.mfs(lemma, pos, w2b_map)
                if synset is None:
                    if mode == 'bn':
                        result[id] = None
                    elif mode == 'dom':
                        result[id] = 'factotum'
                    elif mode == 'lex':
                        result[id] = 'adj.all'
                else:
                    num_mfs += 1
                    if mode == 'bn':
                        result[id] = synset
                    elif mode == 'dom':
                        if synset in bn2sense_map:
                            result[id] = bn2sense_map[synset]
                        else:
                            result[id] = 'factotum'
                    elif mode == 'lex':
                        if synset in bn2sense_map:
                            result[id] = bn2sense_map[synset]
                        else:
                            result[id] = 'adj.all'

    print("MFS SU TOT WORDS: ", num_mfs, " / ", num_words)

    print("write result on file")
    # HERE WRITE THE RESULT ON THE OUTPUT PATH
    write_result_on_file(output_path, result)
    print("create file in:", output_path)
    print("Done!")


def predict_on_all():
    '''
    Method used as procedure to predict all of all the evaluation datasets.
    :return: None
    '''
    dataset_names = ["ALL","semeval2007", "semeval2013","semeval2015","senseval2","senseval3"]
    b2dom, dom2b = par.create_maps_from_tsv(gp.resources_path + gp.b2wn_domains_map_path)
    b2lex, lex2b = par.create_maps_from_tsv(gp.resources_path + gp.b2lexnames_map_path)
    for name in dataset_names:
        print("predict on dataset "+name+".data.xml ...")
        predict_babelnet(gp.resources_path + gp.test_dir + name+".data.xml",
                         gp.output_dir+name+".babelnet.output.txt",
                         gp.resources_path)

        predict_wordnet_domains(gp.resources_path + gp.test_dir + name+".data.xml",
                                gp.resources_path +gp.output_dir+name+".wndomains.output.txt",
                                gp.resources_path)

        predict_lexicographer(gp.resources_path + gp.test_dir + name+".data.xml",
                              gp.resources_path +gp.output_dir+name+".lexnames.output.txt",
                              gp.resources_path)

        predict_MFS(gp.resources_path+ gp.test_dir + name +".data.xml",
                       gp.resources_path +gp.output_dir+name+".mfs.output.txt",
                       gp.resources_path,None,mode='bn')

        predict_MFS(gp.resources_path + gp.test_dir + name + ".data.xml",
                    gp.resources_path +gp.output_dir + name + ".mfs.output.txt",
                    gp.resources_path, b2dom, mode='dom')

        predict_MFS(gp.resources_path + gp.test_dir + name + ".data.xml",
                    gp.resources_path +gp.output_dir + name + ".mfs.output.txt",
                    gp.resources_path, b2lex, mode='lex')

        print("Done!")



