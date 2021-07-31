## Prerequisites
* python version - python3.7* or higher
* from inside `question_classifier` folder install needed libraries by running
```bash
pip install -r requirements.txt
```


## Running the model

### Training neural network

In order to train the neural network classifier, the following command in command line from inside `src` directory should be executed:
```bash
python3 question_classifier.py --train --config [configuration file path]
```
where `configuration file path` is a path to a file, where configuration of neural network is set up (Check **Configuration setup** section on how to set up custom configuration).

Running this will create and save the model in the data folder as well as create file with evaluation of the training process.

### Testing neural network
In order to test the neural network classifier, the following command in command line from inside `src` directory should be executed:
```bash
python3 question_classifier.py --test --config [configuration file path]
```
where `configuration file path` is a path to a file, where configuration of the model is set up (Check **Configuration setup** section on how to set up custom configuration).

Running this will create file with testing results, such as accuracy and loss of predictions.

### Current configuration
There are two configuration files: bow.config, bilstm.config. Each of them refers to their own model, BOW or BiLSTM, and they can be used to train and test the model over specified configuration.
Their data path (../data/bow.config, ../data/bilstm.config) can be put in place after --config when running the program as specified above.

## Configuration setup
In order to run custom configuration of neural network, user needs to create a configuration file inside data folder.

### Configuration template

```ini
[PATH]
path_train = <train_dataset> - path to training dataset
path_dev = <dev_dataset> - path to dev dataset used for validation and tuning
path_test = <test_dataset> = path to test dataset 

[MODEL_OPTIONS]
model = <model_type> - type of model
path_model = <path_model> - path to the file where model is saved to/loaded from

[HYPERPARAMETERS]
epochs = <epochs> - number of epochs to train for
lowercase = <lowercase> - if to lowercase words in data
lr = <lr> - learning rate for SGD

[EMBEDDING]
embedding = <embedding> - either 'random' for random embeddings, or contains data path with pretrained embeddings
freeze = <freeze> - parameter which allow/forbids fine-tuning (should be specified for both type of embeddings, but will not be used with random)

[NN_STRUCTURE]
word_embedding_dim = <dim> - dimension of the embedding vector
batch_size = <batch> - size of minibatch for batch SGD
hidden_dim = <hidden_dim> - dimension of hidden layer in NN.
earlystopping = <earlystopping> - parameter to stop learning after certain number of epochs if underperforming

[EVALUATION]
path_eval = <path_eval> - path for evaluation results
path_test = <path_test> - path for testing results
```

### Notes
When using `pretrained embeddings`, make sure that dimension of vectors in preloaded text file is equal to parameter `word_embedding_dim` in `[NN_STRUCTURE]`.
All the data files that will be created/used during the training/testing should go inside `data` folder.
Data should be whitespace separated where first comes label and then the question.