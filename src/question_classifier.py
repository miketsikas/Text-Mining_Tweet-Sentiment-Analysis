import torch as pt
import random
import argparse
import data_preprocessing as dp
from config_parser import ConfigEntity
import neural_network as nn

pt.manual_seed(1)
random.seed(1)


# adding parsing arguments for convenience and code readability
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Configuration file')
parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')

# test() function for testing the model
# ARGUMENTS:
# -- config_file - configuration file with info  about the model to execute testing on
# RETURNS:
# -- output.txt - file with resulting classifications of test data as well as accuracy of classification
def test(config_file):
    config = ConfigEntity(config_file)
    paths = config.get_path()
    model = config.get_model()
    hyperparams = config.get_hyperparameters()
    evaluation = config.get_evaluation()

    data = open('../data/word2idx', 'r').readlines()
    word2idx = {i.split(' ')[0]: int(i.split(' ')[1]) for i in data}
    data = open('../data/label2idx', 'r').readlines()
    label2idx = {i.split(' ')[0]: int(i.split(' ')[1]) for i in data}

    model_path = model['path_model']
    results_path = evaluation['path_test']
    test_input, test_labels, max_length = dp.data_load(paths['path_train'], hyperparams['lowercase'] == 'True')
    test_input = dp.encode_data(test_input, word2idx, max_length)
    test_labels = dp.encode_labels(test_labels, label2idx)
    nn.test(model_path, test_input, test_labels, results_path, label2idx.keys())
    pass

# train() function for training the model
# ARGUMENTS:
# -- config_file - configuration file with info about the future model to train
# RETURNS:
# -- trained classifier based on model configuration
def train(config_file):
    config = ConfigEntity(config_file)
    paths = config.get_path()
    model = config.get_model()
    hyperparams = config.get_hyperparameters()
    embedding = config.get_embedding()
    pretrained_emb = None

    nn_structure = config.get_nn_structure()
    evaluation = config.get_evaluation()
    evaluation_file = evaluation['path_eval']
    model_file = model['path_model']
    data_unparsed_input, data_labels, max_length = dp.data_load(paths['path_train'], hyperparams['lowercase'] == 'True')
    word_dict = dp.create_dictionary(data_unparsed_input)
    classes = dp.retrieve_unique_classes(data_labels)
    word2idx = dp.word_to_idx_mapping(word_dict)
    label2idx = dp.labels_to_idx_mapping(classes)
    processed_input = dp.encode_data(data_unparsed_input, word2idx, max_length)
    processed_labels = dp.encode_labels(data_labels, label2idx)
    valid_input, valid_labels, val_length = dp.data_load(paths['path_dev'], hyperparams['lowercase'] == 'True')
    valid_input = dp.encode_data(valid_input, word2idx, val_length)
    valid_labels = dp.encode_labels(valid_labels, label2idx)
    if embedding['embedding'] != 'random':
        pretrained_emb = dp.retrieve_embeddings(embedding['embedding'], word_dict,
                                                int(nn_structure['word_embedding_dim']))
    nn.train(model['model'], processed_input, processed_labels, valid_input, valid_labels, int(nn_structure['hidden_dim']),
             int(nn_structure['word_embedding_dim']), len(classes),
             len(word_dict), int(nn_structure['batch_size']), float(hyperparams['lr']),
             int(hyperparams['epochs']), evaluation_file, model_file, pretrained_emb, embedding['freeze'] == 'True',
             int(nn_structure['earlystopping']))
    with open('../data/word2idx', 'w') as file:
        for word in word2idx:
            file.write(word + ' ' + str(word2idx[word]) + '\n')

    with open('../data/label2idx', 'w') as file:
        for label in label2idx:
            file.write(label + ' ' + str(label2idx[label]) + '\n')


args = parser.parse_args()
if args.train:
    # call train.txt function
    train(args.config)
elif args.test:
    # call test function
    test(args.config)
