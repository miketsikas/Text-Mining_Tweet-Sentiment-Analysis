import torch as pt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def train(model_type, processed_input, processed_labels, valid_input, valid_labels, hidden_dim, vector_dim,
          num_labels, vocab_size, batch_size, lr, epochs, evaluation_file, model_file, pretrained_emb=None,
          freeze=True, earlystopping=10):
    if pt.cuda.is_available():
        device = pt.device("cuda")
        print(f'There are {pt.cuda.device_count()} GPU(s) available.')
        print('Device name:', pt.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = pt.device("cpu")

    if model_type == 'bow':
        model = BOWFFNN(hidden_dim, num_labels, vector_dim, vocab_size, pretrained_emb, freeze)
    else:
        model = LSTMFFNN(hidden_dim, num_labels, vector_dim, vocab_size, pretrained_emb, freeze)
    train_dataset = QuestionsDataset(processed_input, processed_labels)
    valid_dataset = QuestionsDataset(valid_input, valid_labels)
    train_data_loaded = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loaded = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    loss_function = pt.nn.CrossEntropyLoss()
    optimizer = pt.optim.SGD(model.parameters(), lr=lr)
    file = open(evaluation_file, 'w')
    file.write('Evaluation during training.\n')
    val_loss_list = []
    epochs_no_improve = 0
    for epoch in range(epochs):
        start_time = time.time()
        sum_loss = 0.0
        total = 0
        early_stop = True
        for x, y in train_data_loaded:
            optimizer.zero_grad()
            x = x.long()
            y = y.long()
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]

        val_loss, val_acc = metrics(model, train_data_loaded)
        val_loss_list.append(val_loss)
        # If the validation loss is at a minimum
        if val_loss > min(val_loss_list):
            # torch.save(model) # Save the model
            epochs_no_improve += 1
        else:
            epochs_no_improve = 0

        # Check early stopping condition
        if early_stop and epoch > 5 and epochs_no_improve == earlystopping:
            print('Early stopping!')
            print("Stopped")
            break

        elapsed_time = time.time() - start_time
        spacing = '|=================================================================================================================|'
        errors = 'Epoch {}/{} \t Time={:.2f}s \t Train Loss={:.4f} \t Validation Loss={:.4f}  \t Validation Accuracy={:.4f}'.format(
                epoch + 1, epochs, elapsed_time, sum_loss / total, val_loss, val_acc)
        print(spacing)
        print(errors)
        file.write(spacing + '\n')
        file.write(errors + '\n')

    file.write('\nValidation.\n')
    valid_loss, valid_acc = metrics(model, valid_data_loaded)
    errors = 'Loss={:.4f}  \tAccuracy={:.4f}'.format(valid_loss, valid_acc)
    file.write(errors + '\n')
    file.close()
    pt.save(model, model_file)
    print('Model file path:', model_file)
    print('Evaluation file path:', evaluation_file)

def metrics(model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    for x, y in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        pred = pt.max(y_hat, 1)[1]
        correct += pt.sum(pred==y)
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
    return sum_loss/total, correct/total


def test(model_path, input, labels, test_results_file, classes):
    model = pt.load(model_path)
    input = pt.LongTensor(input)
    labels = pt.LongTensor(labels)
    y_pred = model(input)
    loss = F.cross_entropy(y_pred, labels)
    pred = pt.max(y_pred, 1)[1]
    correct = pt.sum(pred == labels)
    total = labels.shape[0]
    sum_loss = loss.item() * labels.shape[0]
    acc = (correct / total).item()
    loss = sum_loss / total
    with open(test_results_file, 'w') as file:
        file.write('TESTING RESULTS\n')
        file.write('ACCURACY: ' + str(acc) + '\n')
        file.write('LOSS: ' + str(loss) + '\n')
        report = classification_report(pred, labels, target_names=classes, zero_division=0)
        file.write('TABLED REPORT\n')
        file.write(report)
    print('Testing results file path:', test_results_file)

class BOWFFNN(pt.nn.Module):
    def __init__(self, hidden_dim, output_dim, embedding_dim, vocab_size, pretrained_embeddings=None, freeze=True):
        super(BOWFFNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.vector_dim = embedding_dim
        if pretrained_embeddings is not None:
            self.embedding = pt.nn.Embedding.from_pretrained(pt.from_numpy(pretrained_embeddings), freeze)
        else:
            self.embedding = pt.nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = pt.nn.Linear(self.vector_dim, self.hidden_dim)
        self.fc2 = pt.nn.Linear(self.hidden_dim, self.output_dim)

    def bow_word_embeddings(self, x):
        emb = self.embedding(x)
        return emb.sum(axis=1) / len(emb)

    def forward(self, x):
        emb = self.bow_word_embeddings(x)
        hidden = self.fc1(emb)
        output = self.fc2(hidden)
        m = pt.nn.LogSoftmax(dim=0)

        return m(output)


class LSTMFFNN(pt.nn.Module):
    def __init__(self, hidden_dim, output_dim, embedding_dim, vocab_size, pretrained_embeddings=None, freeze=True):
        super(LSTMFFNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.vector_dim = embedding_dim
        if pretrained_embeddings is not None:
            print('Pretrained embeddings')
            self.embedding = pt.nn.Embedding.from_pretrained(pt.from_numpy(pretrained_embeddings), freeze)
        else:
            print('Random embeddings')
            self.embedding = pt.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = pt.nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.linear = pt.nn.Linear(hidden_dim, output_dim)
        self.dropout = pt.nn.Dropout(0.2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])


class QuestionsDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return pt.from_numpy(self.x[i].astype(np.int32)), pt.tensor(self.y[i])
