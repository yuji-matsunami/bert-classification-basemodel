from torch._C import device
from torch.optim import adagrad, optimizer
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

def train(model, device, EPOCH, train_dataset, test_dataset, optimizer):
    losses = []
    # modelを学習モードにする
    model.train()
    # datasetをミニバッチのイテレーションに変換
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    for epoch in range(EPOCH):
        print(f"-------epoch:{epoch}-----")
        for batch in train_loader:
            # 最適化関数の初期化
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            print(f"loss:{loss}")
            losses.append(loss)
            loss.backward()
            optimizer.step()

    """
    test(val)
    """
    # modelを検証モードにする
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            outputs = model(input_ids)
            pred = torch.argmax(outputs['logits'], axis=1).item()
            print(f"pred:{pred}")
            print(f"label:{batch['label']}")
            if pred == batch['label']:
                correct += 1
    # optunaは誤り率を返す必要がある
    acc = correct/len(test_dataset)
    print(f"acc:{acc}")
    return 1-acc
        
        
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

def objective(trial):
    global train_dataset, EPOCH, eval_dataset
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-v2')
    model.to(device)
    optimizer = get_optimizer(trial, model)
    err_rate = train(model, device, EPOCH, train_dataset, eval_dataset, optimizer)
    return err_rate


def main():
    global train_dataset, EPOCH, eval_dataset
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    yml = read_yml()
    data_dir = yml['data_dir']
    EPOCH = yml['epoch']
    TRIAL_SIZE = yml['trial_size']
    dataset = DataReader(dir_name=data_dir)
    train_df, test_df = dataset.binary_classfication_dataset()
    train_encodings, train_labels, val_encodings, val_labels = transform(train_df, train=True)
    test_encodings, test_lables = transform(test_df)
    train_dataset = ABSADataset(train_encodings, train_labels)
    eval_dataset = ABSADataset(val_encodings, val_labels)
    test_dataset = ABSADataset(test_encodings, test_lables)
    print(f"train_encodings:{len(train_encodings['input_ids'])}")
    print(f"testencodings:{len(test_encodings['input_ids'])}")
    print(f"test_dataset:{len(test_dataset)}")
    # study = optuna.create_study()
    # study.optimize(objective, n_trials=TRIAL_SIZE)
    model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-v2')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    train(model,device,EPOCH, train_dataset, test_dataset, optimizer)
    



if __name__=="__main__":
    main()