from torch.optim import adagrad
import yaml
from make_dataset import DataReader,test_binary_dataset
from transform import transform
from transformers import BertForSequenceClassification
import torch
from torch.utils.data import DataLoader
from dataset import ABSADataset
import optuna
import torch.optim as optim

def read_yml():
    """[summary]
    - Load the learning configuration file
    Returns:
        [type]dict: [description] 
    """
    with open("config.yaml", 'r') as f:
        yml = yaml.load(f)
    return yml

def train(model, device, EPOCH, train_dataset, optimizer):
    losses = []
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    for epoch in range(EPOCH):
        print("-------epoch:{%d}-----".format(epoch))
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            losses.append(loss)
            loss.backward()
            optimizer.step()

        
def get_optimizer(trial, model):
    optimizer_names = ['Adam', 'Adagrad', 'SGD']
    optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)
    weight_decay = trial.suggest_loguniform('wright_decay', 1e-10, 1e-3)
    if optimizer_name == 'Adam':
        adam_lr = trial.suggest_loguniform('adma_lr', 1e-5, 1e-1)
        optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adagrad':
        adagrad_lr = trial.suggest_loguniform('adagrad_lr', 1e-5, 1e-1)
        optimizer = optim.Adagrad(model.parameters(), lr=adagrad_lr, weight_decay=weight_decay)
    else:
        sgd_lr = trial.suggest_loguniform('sgd_lr', 1e-5, 1e-1)
        optimizer = optim.SGD(model.parameters(), lr=sgd_lr, weight_decay=weight_decay)
    return optimizer
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    yml = read_yml()
    data_dir = yml['data_dir']
    EPOCH = yml['epoch']
    dataset = DataReader(dir_name=data_dir)
    train_df, test_df = dataset.binary_classfication_dataset()
    train_encodings, train_labels, val_encodings, val_labels = transform(train_df, train=True)
    test_encodings, test_lables = transform(test_df)
    train_dataset = ABSADataset(train_encodings, train_labels)
    model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-v2')
    model.to(device)
    train(model, device, EPOCH, train_dataset)



if __name__=="__main__":
    main()