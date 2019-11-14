#FILE THAT CONTAINS ONLY PATHS USED IN THE PROJECT

#DIRECTORY PATHS
resources_path = "../resources/"
train_dir = "train/"
dev_dir = "dev/"
test_dir = "test/"
output_dir = "output/"
dataset_dir = "dataset_xml/"
map_dir = "mapping/"


#SEMCOR
semCor_path = train_dir+"semcor.data.xml"
semCor_train_input_path = train_dir+"semcor.input.txt"
semCor_train_bn_labels_path = train_dir + "semcor.bn.labels.txt"
semCor_train_dom_labels_path = train_dir + "semcor.dom.labels.txt"
semCor_train_lex_labels_path = train_dir + "semcor.lex.labels.txt"
semCor_gold_key_path = train_dir+"semcor.gold.key.txt"


#TRAIN-ON-MATIC
tom_path = train_dir+"train-o-matic.xml"
tom_input_path = train_dir+"tom.input.txt"
tom_resized_input_path = train_dir+"tom.resize.input.txt"
tom_bn_labels_path = train_dir+"tom.bn.labels.txt"
tom_resized_bn_labels_path = train_dir+"tom.resize.bn.labels.txt"
tom_dom_labels_path = train_dir+"tom.dom.labels.txt"
tom_lex_labels_path = train_dir+"tom.lex.labels.txt"

#SEMEVAL2007
semeval2007_path = dev_dir+"semeval2007.data.xml"
semeval2007_dev_input_path = dev_dir+"semeval2007.input.txt"
semeval2007_dev_bn_labels_path = dev_dir+"semeval2007.bn.labels.txt"
semeval2007_dev_dom_labels_path = dev_dir+"semeval2007.dom.labels.txt"
semeval2007_dev_lex_labels_path = dev_dir+"semeval2007.lex.labels.txt"
semeval2007_gold_key_path = dev_dir+"semeval2007.gold.key.txt"


#TEST  (EVALUATION DATASETS)

#ALL
all_data_test_path = test_dir + "ALL.data.xml"
all_data_test_key_path = test_dir+ "ALL.gold.key.txt"

#SEMEVAL 2013
semeval2013_test_path = test_dir + "semeval2013.data.xml"
semeval2013_test_key_path = test_dir + "semeval2013.gold.key.txt"

#SEMEVAL 2015
semeval2015_test_path = test_dir + "semeval2015.data.xml"
semeval2015_test_key_path = test_dir + "semeval2015.gold.key.txt"

#SENSEVAL2
senseval2_test_path = test_dir + "senseval2.data.xml"
senseval2_test_key_path = test_dir + "senseval2.gold.key.txt"

#SENSEVAL3
senseval3_test_path = test_dir + "senseval3.data.xml"
senseval3_test_key_path = test_dir + "senseval3.gold.key.txt"


#MAPPING
b2w_map_path = "babelnet2wordnet.tsv"
b2wn_domains_map_path = "babelnet2wndomains.tsv"
b2lexnames_map_path = "babelnet2lexnames.tsv"
