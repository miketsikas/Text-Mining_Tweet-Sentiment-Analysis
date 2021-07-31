import configparser


class ConfigEntity:
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.sections()
        config.read(config_path)

        path = config['PATH']
        self.path = path
        model = config['MODEL_OPTIONS']
        self.model = model
        hyperparameters = config['HYPERPARAMETERS']
        self.hyper_parameters = hyperparameters
        embeddings = config['EMBEDDING']
        self.embedding_options = embeddings
        nn_structure = config['NN_STRUCTURE']
        self.nn_structure = nn_structure
        evaluation = config['EVALUATION']
        self.evaluation = evaluation

    def get_path(self):
        return self.path

    def get_model(self):
        return self.model

    def get_hyperparameters(self):
        return self.hyper_parameters

    def get_embedding(self):
        return self.embedding_options

    def get_nn_structure(self):
        return self.nn_structure

    def get_evaluation(self):
        return self.evaluation
