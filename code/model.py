import string

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import ssl
import preprocessing as pre

from tensorflow.python.keras import backend as K, metrics
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Flatten, RepeatVector, Activation, Permute, multiply,Lambda, TimeDistributed
from tensorflow.python.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Dropout,Concatenate,Multiply
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV



def ELMoEmbedding(x):
    '''
    Method that implements the ELMO Embeddings using Tensorflow Hub Module.
    :param x the input data
    :return the embedding of the input
    '''
    try:
        url = "https://tfhub.dev/google/elmo/2"
        print("Donwload Elmo module from "+url+" ... (Required several minutes!)")

        embed = hub.Module(url, trainable=True)
        print("Elmo module downloaded!")
    except ssl.SSLError as err:
        print("Error: ", err)
    return embed(tf.reshape(tf.cast(x, tf.string),[-1]), signature="default", as_dict=True)["elmo"]



def create_2_BLSTM_layers(vocab_input_data_size, output_bn_vocab_size, output_dom_vocab_size,
                           output_lex_vocab_size, embedding_size, hidden_size, input_dropout, lstm_dropout):
    '''
    Method used to create a double BLSTM model
    :param vocab_input_data_size the size of the input vocab
    :param output_bn_vocab_size the size of the output bn vocab
    :param output_dom_vocab_size the size of the output dom vocab
    :param output_lex_vocab_size the size of the output lex vocab
    :param embedding_size the size of the Keras Embedding
    :param hidden_size the deep of the LSTM Layer
    :param input_dropout indicates how much dropout after the input Layer
    :param lstm_dropout indicates how much dropout in the LSTM Layer
    :return a tensorflow.keras.model
    '''

    print("Creating 2 BLSTM multitask")

    input_data = Input(shape=(None,))

    x1 = Embedding(vocab_input_data_size, embedding_size, mask_zero=True)(input_data)

    dropout = Dropout(input_dropout, name="Dropout")(x1)

    layer1_bidirectional = Bidirectional(
        LSTM(hidden_size, return_sequences=True, recurrent_dropout=lstm_dropout,
             dropout=lstm_dropout))(dropout)
    
    layer2_bidirectional = Bidirectional(
        LSTM(hidden_size, return_sequences=True, recurrent_dropout=lstm_dropout,
             dropout=lstm_dropout))(layer1_bidirectional)
    
    bn_output = Dense(output_bn_vocab_size, activation='softmax',name='bn_output')(layer2_bidirectional)
    dom_output = Dense(output_dom_vocab_size, activation='softmax',name='dom_output')(layer2_bidirectional)
    lex_output = Dense(output_lex_vocab_size, activation='softmax',name='lex_output')(layer2_bidirectional)

    model = Model(inputs=input_data, outputs=[bn_output,dom_output,lex_output])
    return model


def create_BLSTM(vocab_input_data_size, output_bn_vocab_size, output_dom_vocab_size,
                                           output_lex_vocab_size, embedding_size, hidden_size, input_dropout, lstm_dropout,elmo=False,attention = False):
    '''
    Method used to create a single BLSTM model with additional layers extra like Elmo Embeddings and Attention Layer.
    :param vocab_input_data_size the size of the input vocab
    :param output_bn_vocab_size the size of the output bn vocab
    :param output_dom_vocab_size the size of the output dom vocab
    :param output_lex_vocab_size the size of the output lex vocab
    :param embedding_size the size of the Keras Embedding
    :param hidden_size the deep of the LSTM Layer
    :param input_dropout indicates how much dropout after the input Layer
    :param lstm_dropout indicates how much dropout in the LSTM Layer
    :param elmo boolean that indicates if you want to use Elmo embeddings
    :param attention boolean that indicates if you want to use the Attention Layer
    :return a tensorflow.keras.model
    '''
    if attention and elmo:
        print("Creating BLSTM with ELMO and Attention Layer")
    elif attention and not  elmo:
        print("Creating BLSTM with Attention Layer")
    elif not attention and elmo:
        print("Creating BLSTM with ELMO")
    else:
        print("Creating BLSTM ")

    if not elmo:
        input_data = Input(shape=(None,))
        x1 = Embedding(vocab_input_data_size, embedding_size, mask_zero=True)(input_data)
    else:
        input_data = Input(shape=(None,), dtype=tf.string)
        x1 = Lambda(ELMoEmbedding)(input_data)

    dropout = Dropout(input_dropout, name="Dropout")(x1)

    lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(
        LSTM(hidden_size, return_sequences=True,return_state=True, recurrent_dropout=lstm_dropout,
             dropout=lstm_dropout))(dropout)

    #PRINCIPALLY I FOUND AN ATTENTION LAYER WITH FLATTEN, REPEAT VECTOR AND PERMUTE,
    #BUT MODIFYING IT WITH ONLY PARTS THAT ARE NOT COMMENTED THE MODEL HAS MORE PERFORMANCE.
    if attention:
        # compute importance for each step
        attention_layer = TimeDistributed(Dense(1, activation='tanh'))(lstm)
        #attention = Flatten()(attention)
        attention_layer = Activation('softmax')(attention_layer)
        #attention_layer = RepeatVector(hidden_size*2)(attention_layer)
        #attention_layer = Permute([2, 1])(attention_layer)

        #cont = Multiply()([lstm,attention_layer])
        #cont = Dense(hidden_size*2,activation='tanh')(cont)

        #merged_output = Concatenate(axis=-1)([lstm,cont])
        merged_output = Multiply()([lstm, attention_layer])
    else:
        merged_output = lstm

    #merged_output = Concatenate([lstm, attention], mode='mul')
    #merged_output = multiply([lstm, attention])

    bn_output = Dense(output_bn_vocab_size, activation='softmax', name='bn_output')(merged_output)
    dom_output = Dense(output_dom_vocab_size, activation='softmax', name='dom_output')(merged_output)
    lex_output = Dense(output_lex_vocab_size, activation='softmax', name='lex_output')(merged_output)
    model = Model(inputs=input_data, outputs=[bn_output, dom_output, lex_output])
    return model




def create_singletask_BLSTM(vocab_input_data_size, output_vocab_size, embedding_size
                            , hidden_size, input_dropout, lstm_dropout):
    '''
       Method used to create a single BLSTM with only one Dense output Layer. No for the Multitask.
       :param vocab_input_data_size the size of the input vocab
       :param output_bn_vocab_size the size of the output bn vocab
       :param output_dom_vocab_size the size of the output dom vocab
       :param output_lex_vocab_size the size of the output lex vocab
       :param embedding_size the size of the Keras Embedding
       :param hidden_size the deep of the LSTM Layer
       :param input_dropout indicates how much dropout after the input Layer
       :param lstm_dropout indicates how much dropout in the LSTM Layer
       :return a tensorflow.keras.model
    '''
    print("Creating BLSTM singletask")

    input_data = Input(shape=(None,))

    x1 = Embedding(vocab_input_data_size, embedding_size, mask_zero=True)(input_data)

    dropout = Dropout(input_dropout,name="Dropout")(x1)


    layer1_bidirectional = Bidirectional(
        LSTM(hidden_size, return_sequences=True, recurrent_dropout=lstm_dropout,
             dropout=lstm_dropout))(dropout)


    output = Dense(output_vocab_size, activation='softmax')(layer1_bidirectional)
    model = Model(inputs=input_data, outputs=output)
    return model



def compile_keras_model(model,optimizer,learning_rate):
    '''
    Method that takes in input a model and compile it using the specific optimizer. The model is compiled
    with sparse_categorical_crossentropy as loss function.
    :param model: the tensorflow.keras.model implemented.
    :param optimizer: there are several options: 1)sgd ; 2)Adam( I used principally for the training)
    3) adadelta
    :param learning_rate: float that indicates the learning rate to use with the optimizer.
    :return: the model compiled.
    '''
    if(str.lower(optimizer) == "sgd"):
        opt = optimizers.SGD(lr=learning_rate, clipnorm=0.1 , momentum=0.95, nesterov=True)
    elif(str.lower(optimizer) == "adam"):
        opt = optimizers.Adam(lr=learning_rate)
    elif(str.lower(optimizer) == "adadelta"):
        opt = optimizers.adadelta(lr=learning_rate)
    sess = K.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=opt, metrics=["acc"])
    return model


def batch_generator_multitask(input_data, bn_labels,dom_labels,lex_labels, batch_size,pad_sequence = False):
    '''
    Method used to fit the input to the multitask model that implements Keras Embeddings in batches.
    When is used the padding, the method find the max length of the bigger sentence inside the batch and compare it
    with a 50, a fixed value used to truncate the samples.
    :param input_data: array of arrays of id tokens taken from output bn vocab
    :param bn_labels: array of arrays of id tokens taken from output bn vocab
    :param dom_labels: arrays of arrays of id tokens taken from the output dom vocab
    :param lex_labels: arrays of arrays of id tokens taken from the output lexnames vocab
    :param batch_size: integer that indicates the size of batch
    :param pad_sequence: boolean that indicates if do padding or not.
    :return: a generator that contains the batch input, and a list of batch bn labels, batch dom labels and batch lex labels
    '''
    while True:
      for start in range(0, len(input_data), batch_size):
          end = start + batch_size
          max_length = pre.find_max_length(input_data[start:end])
          max_length = min(max_length,50)
          batch_input = input_data[start:end]
          batch_bn_labels = bn_labels[start:end]
          batch_dom_labels = dom_labels[start:end]
          batch_lex_labels = lex_labels[start:end]
          if pad_sequence:
              batch_input = pad_sequences(batch_input, maxlen=max_length, padding='post',truncating='post')
              batch_bn_labels = pad_sequences(batch_bn_labels, maxlen=max_length, padding='post',truncating='post')
              batch_dom_labels = pad_sequences(batch_dom_labels, maxlen=max_length, padding='post',truncating='post')
              batch_lex_labels = pad_sequences(batch_lex_labels, maxlen=max_length, padding='post',truncating='post')

          batch_bn_labels = np.reshape(batch_bn_labels,(len(batch_bn_labels), len(batch_bn_labels[0]), 1))
          batch_dom_labels = np.reshape(batch_dom_labels,(len(batch_dom_labels), len(batch_dom_labels[0]), 1))
          batch_lex_labels = np.reshape(batch_lex_labels,(len(batch_lex_labels), len(batch_lex_labels[0]), 1))

          yield batch_input, [batch_bn_labels, batch_dom_labels, batch_lex_labels]


def batch_generator_elmo(input_data, bn_labels,dom_labels,lex_labels, batch_size,pad_sequence = False):
    '''
    Method used to fit the input to the multitask model that implements ELMO Embeddings in batches.
    When is used the padding, the method find the max length of the bigger sentence inside the batch and compare it
    with a 50, a fixed value used to truncate the samples.
    :param input_data: array of arrays of strings where each word is separated by a blank space
    :param bn_labels: array of arrays of id tokens taken from output bn vocab
    :param dom_labels: arrays of arrays of id tokens taken from the output dom vocab
    :param lex_labels: arrays of arrays of id tokens taken from the output lexnames vocab
    :param batch_size: integer that indicates the size of batch
    :param pad_sequence: boolean that indicates if do padding or not.
    :return: a generator that contains the batch input, and a list of batch bn labels, batch dom labels and batch lex labels
    '''
    while True:
        for start in range(0, len(input_data), batch_size):
            end = start + batch_size
            max_length_input = pre.find_max_length_sentence(input_data[start:end])
            max_length_output = pre.find_max_length(bn_labels[start:end])
            max_length_input = min(max_length_input,50)
            max_length_output = min(max_length_output,50)
            batch_input = input_data[start:end]
            batch_bn_labels = bn_labels[start:end]
            batch_dom_labels = dom_labels[start:end]
            batch_lex_labels = lex_labels[start:end]
            if pad_sequence:
                  batch_input = pre.pad_sentence_array(batch_input,max_length_input)
                  batch_bn_labels = pad_sequences(batch_bn_labels, maxlen=max_length_output, padding='post',truncating='post')
                  batch_dom_labels = pad_sequences(batch_dom_labels, maxlen=max_length_output, padding='post',truncating='post')
                  batch_lex_labels = pad_sequences(batch_lex_labels, maxlen=max_length_output, padding='post',truncating='post')

            batch_bn_labels = np.expand_dims(batch_bn_labels, -1)
            batch_dom_labels = np.expand_dims(batch_dom_labels, -1)
            batch_lex_labels = np.expand_dims(batch_lex_labels, -1)

            yield batch_input, [batch_bn_labels, batch_dom_labels, batch_lex_labels]


def batch_generator(input_data, labels, batch_size):
    '''
        Method used to fit the input to the singletask model that implements Keras Embeddings in batches.
        :param input_data: array of arrays of id tokens taken from output bn vocab
        :param labels: array of arrays of id tokens taken from one of the sense vocab
        :param batch_size: integer that indicates the size of batch
        :return: a generator that contains the batch input, and a list of batch bn labels, batch dom labels and batch lex labels
     '''
    while True:
        for start in range(0, len(input_data), batch_size):
            end = start + batch_size
            max_length = pre.find_max_length(input_data[start:end])
            batch_input = input_data[start:end]
            batch_labels = labels[start:end]
            batch_input = pad_sequences(batch_input, maxlen=max_length, padding='post')
            batch_labels = pad_sequences(batch_labels, maxlen=max_length, padding='post')
            batch_labels = np.expand_dims(batch_labels, -1)
            yield batch_input, batch_labels


def train_keras_model(model, input_data,labels,dev_input_data,dev_labels,
                                batch_size,epochs,steps_per_epochs,validation_steps,weights_filename):
    '''
    Method used to train the singletask model implemented and compiled.
    :param model: the singletask model implemented and compiled
    :param input_data: arrays of arrays of id tokens
    :param labels: array of arrays of id tokens
    :param dev_input_data: array of array of id tokens containing data from dev set
    :param dev_labels: array of array of id tokens containing data from dev set
    :param batch_size:  integer that indicates the size of batch
    :param epochs: integer that indicates how much epochs to use to train the model
    :param steps_per_epochs: integer that indicates how much step per epochs to use to train the model
    :param validation_steps: integer that indicates how much validation steps per epoch to yse to validate the model
    :param weights_filename: filepath where save the weights of the model
    :return: the statistics of the model after that the training is completed.
    '''
    early_stopping = EarlyStopping(monitor="val_loss", patience=2)
    checkpointer = ModelCheckpoint(filepath="drive/My Drive/"+weights_filename+".hdf5", monitor='val_loss', verbose=1,
                                   save_best_only=True, mode='min')
    cbk = [early_stopping, checkpointer]

    print("\nStarting training...")
    stats = model.fit_generator(batch_generator(input_data, labels,  batch_size),
                                steps_per_epoch=steps_per_epochs,
                                epochs=epochs,
                                callbacks=cbk,
                                verbose=1,
                                validation_data=batch_generator(dev_input_data, dev_labels,  batch_size),
                                validation_steps=validation_steps)
    print("Training complete.\n")
    return stats


def train_keras_model_multitask(model, input_data,bn_labels, dom_labels, lex_labels,dev_input_data,
                                dev_bn_labels, dev_dom_labels, dev_lex_labels,
                                batch_size,epochs,steps_per_epochs,validation_steps,initial_epoch,weights_filename, pad_sequence = False):
    '''
    Method used to train the multitask model with Keras Embeddings implemented and compiled.
    :param model: the multitask model implemented and compiled
    :param input_data: arrays of arrays of id tokens
    :param bn_labels: array of arrays of id tokens
    :param dom_labels: array of arrays of id tokens
    :param lex_labels: array of arrays of id tokens
    :param dev_input_data: array of array of id tokens containing data from dev set
    :param dev_bn_labels: array of array of id tokens containing data from dev set
    :param dev_dom_labels: array of array of id tokens containing data from dev set
    :param dev_lex_labels: array of array of id tokens containing data from dev set
    :param batch_size:  integer that indicates the size of batch
    :param epochs: integer that indicates how much epochs to use to train the model
    :param steps_per_epochs: integer that indicates how much step per epochs to use to train the model
    :param validation_steps: integer that indicates how much validation steps per epoch to yse to validate the model
    :param initial_epoch : integer that indicates the epoch where start to train the model, for example after that the
    training was stopped.
    :param weights_filename: filepath where save the weights of the model
    :param pad_sequence : boolean that indicates if the batch generators have to pad the input data.
    :return: the statistics of the model after that the training is completed.
    '''
    early_stopping = EarlyStopping(monitor="val_bn_output_loss",patience=2)
    checkpointer = ModelCheckpoint(filepath="drive/My Drive/"+weights_filename+".hdf5", monitor='val_bn_output_loss', verbose=1,save_best_only=True,mode='min')
    cbk = [early_stopping,checkpointer]

    print("\nStarting training...")
    stats = model.fit_generator(batch_generator_multitask(input_data, bn_labels, dom_labels,lex_labels,batch_size,pad_sequence),
                        steps_per_epoch=steps_per_epochs,
                        epochs=epochs,
                        callbacks=cbk,
                        verbose = 1,
                        validation_data=batch_generator_multitask(dev_input_data, dev_bn_labels, dev_dom_labels, dev_lex_labels,batch_size,pad_sequence),
                        validation_steps=validation_steps,
                        initial_epoch=initial_epoch)
    print("Training complete.\n")
    return stats


def train_keras_model_elmo(model, input_data,bn_labels, dom_labels, lex_labels,dev_input_data,
                                dev_bn_labels, dev_dom_labels, dev_lex_labels,
                                batch_size,epochs,steps_per_epochs,validation_steps,weights_filename,pad_sequence = False):
    '''
    Method used to train the multitask model with Elmo Embeddings implemented and compiled.
    :param model: the multitask model implemented and compiled
    :param input_data: arrays of arrays of strings. Each word in the string have to be separated from each other with black space.
    :param bn_labels: array of arrays of id tokens
    :param dom_labels: array of arrays of id tokens
    :param lex_labels: array of arrays of id tokens
    :param dev_input_data: array of array of id tokens containing data from dev set
    :param dev_bn_labels: array of array of id tokens containing data from dev set
    :param dev_dom_labels: array of array of id tokens containing data from dev set
    :param dev_lex_labels: array of array of id tokens containing data from dev set
    :param batch_size:  integer that indicates the size of batch
    :param epochs: integer that indicates how much epochs to use to train the model
    :param steps_per_epochs: integer that indicates how much step per epochs to use to train the model
    :param validation_steps: integer that indicates how much validation steps per epoch to yse to validate the model
    :param weights_filename: filepath where save the weights of the model
    :param pad_sequence : boolean that indicates if the batch generators have to pad the input data.
    :return: the statistics of the model after that the training is completed.
    '''

    early_stopping = EarlyStopping(monitor="val_bn_output_loss",patience=1)
    checkpointer = ModelCheckpoint(filepath="drive/My Drive/"+weights_filename+".hdf5", monitor='val_bn_output_loss', verbose=1, save_best_only=True,mode='min')
    cbk = [early_stopping,checkpointer]

    print("\nStarting training...")
    stats = model.fit_generator(batch_generator_elmo(input_data, bn_labels, dom_labels,lex_labels,batch_size,pad_sequence),
                        steps_per_epoch=steps_per_epochs,
                        epochs=epochs,
                        callbacks=cbk,
                        verbose = 1,
                        validation_data=batch_generator_elmo(dev_input_data, dev_bn_labels, dev_dom_labels, dev_lex_labels,batch_size,pad_sequence),
                        validation_steps=validation_steps)
    print("Training complete.\n")
    return stats



