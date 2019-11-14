from lxml import etree as ET
from tqdm import tqdm
from nltk.corpus import wordnet as wn
import global_paths as gp
import string
import re
from nltk.stem import WordNetLemmatizer

#USED TO CLEAN THE DATA
#chars_to_remove = ".,?!'""-:_()[]{};"
chars_to_remove = string.punctuation


def mfs(lemma,POS,w2b_map):
    '''
    Method that return the most frequent sense from Wordnet when there isn't
    any sense for a given lemma in the vocabulary.
    :param lemma the input lemma used to retrieve the most frequent sense of it
    :param POS the Part of Speech Tag of the lemma
    :param w2b_map dictionary containing the mapping between wordnet and babelnet id
    :return the most frequent sense as babelnet synset. If the lemma is not present
    in Wordnet or there isn't a mapping from wordnet synset to babelnet synset,
    the method return None
    '''
    synset = wn.synsets(lemma, pos=POS)
    #if there isnt a synset with same pos, try to return first synset
    if len(synset) == 0:
        synset = wn.synsets(lemma)
    #if return at least 1 synset
    if len(synset) > 0:
        synset_id = "wn:" + str(synset[0].offset()).zfill(8) + synset[0].pos()
        return w2b_map[synset_id] if synset_id in w2b_map else None
    return None


def return_synset_id(key):
    '''
    Method used to return the wordnet synset_id from a sense_key of a specific word given in input.
    :param key the input sense_key
    :return the synset_id of the sense_key
    '''
    synset = wn.lemma_from_key(key).synset()
    synset_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
    return synset_id


def create_gold_key_map(file):
    '''
    Method used to creates the map of the gold_key_map file
    containing the mapping from the instance_id and the wordnet sense_key
    :param file the file.txt containing the gold keys.
    :return the map that has as key the instance_id and the wordnet sense_key as value
    '''
    map = dict()
    with open(file,'r',encoding='utf-8') as f:
        for line in tqdm(f):
            splitted = line.split(" ")
            map[splitted[0]] = splitted[1].strip()
    f.close()
    return map


def convert_gold_key_file_to_bn(file, file_to_write, w2b_map):
    '''
    Method used to convert the wordnet synset id within gold file in the second column
    in babelnet ids.
    :param file the file.txt to read containing the gold keys
    :param file_to_write where write the results
    :param w2b_map dictionary that maps from wordnet id to babelnet id.
    '''
    with open(file_to_write,'w',encoding='utf-8') as fw:
        with open(file,'r',encoding='utf-8') as fr:
            for sentence in tqdm(fr):
                splitted = sentence.strip().split(" ")
                fw.write(splitted[0]+" "+w2b_map[return_synset_id(splitted[1])]+"\n")
    fw.close()
    fr.close()


def convert_gold_key_file_to_dom(file, file_to_write, w2b_map,b2dom_map):
    '''
    Method used to convert the wordnet synset id within gold file in the second column
    in wordnet domains. If there isn't a corresponding domain for the babelnet synset, write on file
    'factotum' as domain.
    :param file the file.txt to read containing the gold keys
    :param file_to_write where write the results
    :param w2b_map dictionary that maps from wordnet id to babelnet id.
    :param b2dom_map dictionary that maps from babelnet id to wordnet domains
    '''
    with open(file_to_write,'w',encoding='utf-8') as fw:
        with open(file,'r',encoding='utf-8') as fr:
            for sentence in tqdm(fr):
                splitted = sentence.strip().split(" ")
                if w2b_map[return_synset_id(splitted[1])] in b2dom_map:
                    fw.write(splitted[0]+" "+b2dom_map[w2b_map[return_synset_id(splitted[1])]]+"\n")
                else:
                    fw.write(splitted[0]+" "+"factotum"+'\n')
    fw.close()
    fr.close()


def convert_gold_key_file_to_lex(file, file_to_write, w2b_map,b2lex_map):
    '''
        Method used to convert the wordnet id within gold file in the second column
        in wordnet domains
        :param file the file.txt containing the gold keys
        :param file_to_write where write the results
        :param w2b_map dictionary that maps from wordnet id to babelnet id.
        :param b2lex_map dictionary that maps from babelnet id to lexnames
        '''
    with open(file_to_write,'w',encoding='utf-8') as fw:
        with open(file,'r',encoding='utf-8') as fr:
            for sentence in tqdm(fr):
                splitted = sentence.strip().split(" ")
                fw.write(splitted[0]+" "+b2lex_map[w2b_map[return_synset_id(splitted[1])]]+"\n")
    fw.close()
    fr.close()


'''
Method that take in input the label file containing babelnet synsets and convert it
in different labels. Used to create the lexnames labels and domains labels.
:param file_to_read the label file.txt that contains the babelnet synsets
:param file_to_write the file.txt where write the output(lexnames labels,domains labels)
:param map_to_convert dictionary used to convert the babelnet synsets in lexnames or domains
:param w2b_map dictionary used to find most frequent sense when the synset is not present in the
map_to_convert
'''
def convert_label_file(file_to_read,file_to_write,map_to_convert,w2b_map):
    num_words = 0
    num_mfs = 0
    pos = ['n','a','v','r']
    with open(file_to_read,'r',encoding='utf-8') as fr:
        with open(file_to_write,'w',encoding='utf-8') as fw:
            for sentence in tqdm(fr):
                for word in sentence.strip().split(" "):
                    #take only instances
                    if "_bn:" in word:
                        splitted = word.split("_bn:")
                        #if the synset is present in the map, use it else try to find the mfs, else use 'factotum' as unknown word
                        if "bn:"+splitted[1] in map_to_convert:
                            fw.write(map_to_convert["bn:" + splitted[1]] + " ")
                        else:
                            #check the pos of the synset
                            if splitted[1][-1] in pos:
                                synset = mfs(splitted[0],splitted[1][-1], w2b_map)
                            else:
                                synset = None
                            if synset is None or synset not in map_to_convert:
                                fw.write("factotum" + " ")
                            else:
                                fw.write(map_to_convert[synset] + " ")
                                num_mfs+=1
                        num_words += 1
                    else:
                        fw.write(word+" ")
                fw.write('\n')
        fw.close()
    fr.close()
    print("mfs over instances: ",num_mfs," / ",num_words)


def create_maps_from_tsv(file):
    '''
    Method that creates two maps given a file.tsv. Used for the mapping between wordnet-babelnet
    , babelnet-lexnames, babelnet-domains.
    :param file the file.tsv where take the mapping informations
    :return two maps: the direct map and the inverse map
    '''
    map = dict()
    inverse_map = dict()
    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            splitted = line.split("\t")
            map[splitted[0]] = splitted[1].strip()
            inverse_map[splitted[1].strip()] = splitted[0]
    f.close()
    return map, inverse_map


def make_file_txt_from_semcor(file_to_read, file_to_write_data, file_to_write_labels, gold_key_map, w2b_map):
    '''
    Method used to create file.txt with sentences and labels from the semcor file.xml.
    For each instance of the sentences are replaced the instance_id with the babelnet_id.
    :param file_to_read the file.xml to parse
    :param file_to_write_data the file.txt where write the data X results.
    :param file_to_write_labels the file.txt where write the labels Y results.
    :param gold_key_map the dict containing the mapping from the instance_id and the sense_keys
    :param w2b_map the dict containing the mapping from the wordnet synsets id and the babelnet_id
    '''
    row_X = ""
    row_Y = ""
    # define the file.txt where write the result
    with open(file_to_write_labels,'w',encoding='utf-8') as fw_labels:
        with open(file_to_write_data,'w',encoding='utf-8') as fw_data:
            # parse the file xml
            for event, elem in tqdm(ET.iterparse(file_to_read, events={'start', 'end'})):
                if event == 'start':
                    #start to read sentence
                    #check if the tag mode is wf or instance
                    if elem.tag == 'wf' and type(elem.text) == str:
                        #join the words with blank space
                        row_Y += str.lower(elem.text.replace(" ","_").replace("-","_")) + " "
                        row_X += str.lower(elem.text.replace(" ","_").replace("-","_")) + " "
                    elif elem.tag == 'instance' and type(elem.text) == str and type(elem.attrib['lemma']) == str:
                        #from instance_id take the sense_key id, from it take the wordnet synset_id
                        #and finally take the babelnet_id
                        bn_id = w2b_map[return_synset_id(gold_key_map[elem.attrib['id']])]
                        row_Y += str.lower(elem.attrib['lemma']).replace(" ","_")+"_"+bn_id+ " "
                        row_X += str.lower(elem.text.replace(" ","_").replace("-","_")) + " "


                #at the end of the sentence, process the X and Y
                if event == 'end':
                    if elem.tag == 'sentence':

                        #write on files data X and labels Y
                        fw_data.write(row_X+"\n")
                        fw_labels.write(row_Y+"\n")

                        #clean strings for next sentence
                        row_X = ""
                        row_Y = ""
                    elem.clear()
    fw_data.close()
    fw_labels.close()




def make_file_txt_from_tom(file_to_read,file_to_write_data, file_to_write_labels, w2b_map):
    '''
        Method used to create file.txt with sentences and labels from the tom file.xml.
        :param file_to_read the file.xml to parse
        :param file_to_write_data the file.txt where write the data X results.
        :param file_to_write_labels the file.txt where write the labels Y results.
        :param w2b_map the dict containing the mapping from the wordnet synsets id to babelnet synsets
    '''
    sentences = dict()
    get_sentence = None
    result = []
    lemmatizer = WordNetLemmatizer()

    with open(file_to_write_data,'w',encoding='utf-8') as fw:
        for event, elem in tqdm(ET.iterparse(file_to_read, events={'start', 'end'})):
            if event == 'start':
                # take the word to substitute
                if elem.tag == 'answer':
                    wn_offset = str(elem.attrib["senseId"][3:])
                if elem.tag == 'context':
                    sentence = elem.text

                if elem.tag == 'head' and type(elem.text) == str and type(elem.tail) == str and type(sentence) == str:
                    anchor = elem.text
                    sentence += anchor + elem.tail
                    get_sentence = True

            if event == 'end':
                if elem.tag == 'instance' and get_sentence:
                    if not sentences.get(sentence):
                        sentences[sentence] = set()
                        fw.write(sentence.lower() + " \n")
                    if w2b_map.get("wn:"+wn_offset):
                        sentences[sentence].add(anchor +
                                                "|" + '_'.join(
                            lemmatizer.lemmatize(anchor, pos=wn_offset[-1]).lower().split()) + "_" + w2b_map.get(
                            "wn:"+wn_offset))
                    get_sentence = False
            elem.clear()

        # process the sentences
        for s in tqdm(sentences.keys()):
            for l in sentences[s]:
                words = l.split("|")
                s = s.replace(words[0],
                              re.search(r"[A-Z|a-z][a-z]*_?[A-Z|a-z]*[a-z]*\_bn\:[0-9]{8}[a|v|n|r]", words[1]).group(0))

            result.append(s)

        # write the result
        with open(file_to_write_labels, 'w', encoding='utf-8') as f:
            for s in result:
                f.write(s.lower() + "\n")
            del result
            f.close()
        fw.close()


def resize_files(file_data_to_read,file_labels_to_read,file_to_write_data,file_to_write_labels,num_of_sentences):
    '''
    Method used to resize the big files like eurosense and tom. It reduces the number of sentences inside the file.
    :param file_data_to_read the file txt containing the train sentences
    :param file_labels_to_read the file txt containing the labels
    :param file_to_write_data the file txt where write the number of sentences chosen
    :param file_to_write_labels the file txt where write the number of labels chosen
    :param num_of_sentences integer that indicates how many sentences write in the output file.
    '''
    #resize files
    for file_w, file_r in [(file_to_write_data, file_data_to_read), (file_to_write_labels, file_labels_to_read)]:
        i = 0
        with open(file_w,'w',encoding='utf-8') as fwd:
            with open(file_r,'r',encoding='utf-8') as frd:
                for sentence in tqdm(frd):
                    if i < num_of_sentences:
                        fwd.write(sentence)
                        i+=1
                    else:
                        break
                frd.close()
            fwd.close()


def parse_test_data(file_to_read):
    '''
    Method used to parse the test data file.xml
    It creates an array of arrays of word that can be
    both wf and instances. If they are instances they are written
    with the follow format : word|id_term|position|lemma in sentence|POS
    :param file_to_read the file.xml where read the data
    :return an array of arrays composed by words that can be wf or
    instances
    '''
    sentences = []
    # parse the file xml
    instances = 0
    for event, elem in tqdm(ET.iterparse(file_to_read, events={'start', 'end'})):
        if event == 'start':
            # start to read sentence
            if elem.tag == "sentence":
                i = -1
                row = []
            # check if the tag mode is wf or instance
            elif elem.tag == 'wf' and type(elem.text) == str:
                i += 1
                row.append(str.lower(elem.text.replace(" ", "_").replace("-","_")) )

            elif elem.tag == 'instance':
                #increment index and append in row: word|id|position(i)|lemma|POS
                i += 1
                word = str.lower(elem.text.replace(" ", "_").replace("-","_")) if type(elem.text) == str else ""
                row.append(word
                           +"|"+elem.attrib['id']
                           +"|"+str(i)
                           +"|"+elem.attrib['lemma']
                           +"|"+elem.attrib['pos'])
                #write the gold
                instances+=1

        # at the end of the sentence, process the X and Y
        if event == 'end':
            if elem.tag == 'sentence':
                sentences.append(row)
    print("numero di instances ",instances)
    return sentences


def convert_gold_key_on_all(w2b_map, bn2lexnames_map,bn2domains_map):
    '''
    Method used as procedure in the main to convert all gold key files of various test dataset
    in gold babelnet, gold domains, gold lexnames
    :param w2b_map the dictionary that contains the map between wordnet synset and babelnet synset
    :param bn2lexnames_map the dictionary that contains the map between babelnet synset and lexnames
    :param bn2domains_map the dictionary that contains the map between babelnet synset and domains
    '''
    dataset_names = ["ALL", "semeval2007", "semeval2013", "semeval2015", "senseval2", "senseval3"]
    for name in dataset_names:
        convert_gold_key_file_to_bn(gp.resources_path + gp.test_dir + name+".gold.key.txt",
                                    gp.resources_path + gp.test_dir + name+".gold.babelnet.txt",
                                    w2b_map)
        convert_gold_key_file_to_dom(gp.resources_path + gp.test_dir +name+ ".gold.key.txt",
                                     gp.resources_path + gp.test_dir + name+".gold.domains.txt",
                                     w2b_map, bn2domains_map)
        convert_gold_key_file_to_lex(gp.resources_path + gp.test_dir +name+ ".gold.key.txt",
                                     gp.resources_path + gp.test_dir +name+ ".gold.lexnames.txt",
                                     w2b_map, bn2lexnames_map)


def main():

    print("Create maps from babelnet to wordnet, domains, lexnames..")
    # CREATE W2B AND B2W MAPS
    b2w, w2b = create_maps_from_tsv(gp.resources_path+gp.b2w_map_path)
    bn2domains, domains2bn = create_maps_from_tsv(gp.resources_path+gp.b2wn_domains_map_path)
    bn2lexnames, lexnames2bn = create_maps_from_tsv(gp.resources_path+gp.b2lexnames_map_path)
    print("Done!")

    # CREATE MAP FROM INSTANCE_ID TO SENSE_KEY
    print("Create maps from instance_id to sense_key..")
    semCor_gold_key_map = create_gold_key_map(gp.resources_path+gp.semCor_gold_key_path)
    semeval2007_gold_key_map = create_gold_key_map(gp.resources_path+gp.semeval2007_gold_key_path)
    print("Done!")

    # GENERATE FILE.TXT FROM THE FILE XML (TRAINING DATA AND LABELS)
    print("Generate files from "+gp.resources_path + gp.semCor_path+" ...")
    make_file_txt_from_semcor(gp.resources_path + gp.semCor_path,
                              gp.resources_path + gp.semCor_train_input_path,
                              gp.resources_path + gp.semCor_train_bn_labels_path,
                              semCor_gold_key_map, w2b)
    print("Done! Generated following files:"
          +gp.resources_path + gp.semCor_train_input_path,"\n",
            gp.resources_path + gp.semCor_train_bn_labels_path)

    print("Generate files from "+gp.resources_path+gp.semeval2007_path)
    # GENERATE FILE.TXT FROM THE FILE XML (DEV DATA AND LABELS)
    make_file_txt_from_semcor(gp.resources_path + gp.semeval2007_path,
                              gp.resources_path + gp.semeval2007_dev_input_path,
                              gp.resources_path + gp.semeval2007_dev_bn_labels_path,
                              semeval2007_gold_key_map, w2b)
    print("Done ! Generated following files:"
          + gp.resources_path+gp.semeval2007_dev_input_path,"\n",
                           gp.resources_path+gp.semeval2007_dev_bn_labels_path)

    # GENERATE FILE.TXT FROM TOM
    print("Generate file from "+gp.resources_path + gp.tom_path," ...")
    make_file_txt_from_tom(gp.resources_path + gp.tom_path,
                           gp.resources_path + gp.tom_input_path,
                           gp.resources_path + gp.tom_bn_labels_path,
                           w2b)

    print("Done! Generated following files:"
          + gp.resources_path + gp.tom_input_path,"\n",
                           gp.resources_path + gp.tom_bn_labels_path)

    #RESIZE THE TOM FILEs
    sentence_tom = 200000
    print("Resize the "+gp.resources_path + gp.tom_input_path,gp.resources_path + gp.tom_bn_labels_path,"\n",
          "manteined only "+str(sentence_tom)," sentences!")
    resize_files(gp.resources_path + gp.tom_input_path,
                 gp.resources_path + gp.tom_bn_labels_path,
                 gp.resources_path + gp.tom_resized_input_path,
                 gp.resources_path + gp.tom_resized_bn_labels_path, sentence_tom)
    print("Done!")


    print("Convert all label files...")
    #CREATE LEXNAMES AND DOMAINS LABELS AFTER BABELNET LABELS FILE CONVERSION
    convert_label_file(gp.resources_path+gp.semCor_train_bn_labels_path,
                       gp.resources_path+gp.semCor_train_dom_labels_path,bn2domains,w2b)

    convert_label_file(gp.resources_path+gp.semCor_train_bn_labels_path,
                       gp.resources_path+gp.semCor_train_lex_labels_path,bn2lexnames,w2b)

    convert_label_file(gp.resources_path + gp.semeval2007_dev_bn_labels_path,
                       gp.resources_path + gp.semeval2007_dev_dom_labels_path, bn2domains,w2b)

    convert_label_file(gp.resources_path + gp.semeval2007_dev_bn_labels_path,
                       gp.resources_path + gp.semeval2007_dev_lex_labels_path, bn2lexnames,w2b)


    #FOR TOM
    convert_label_file(gp.resources_path + gp.tom_resized_bn_labels_path,
                       gp.resources_path + gp.tom_dom_labels_path,
                       bn2domains, w2b)

    convert_label_file(gp.resources_path + gp.tom_resized_bn_labels_path,
                       gp.resources_path + gp.tom_lex_labels_path,
                       bn2lexnames, w2b)
    print("Done!")

    # CONVERT THE GOLD KEY TEST IN BABELNET FORMAT(ALL)
    print("Convert gold key files...")
    convert_gold_key_on_all(w2b,bn2lexnames,bn2domains)
    print("Done!")

if __name__ == "__main__":
    main()
