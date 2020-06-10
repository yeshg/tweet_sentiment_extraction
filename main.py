from utils import *
import pandas as pd

import os

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import hashlib
import time

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, data, word_to_ix, max_len=102, test_dset=False):
        self.data = data
        self.word_to_ix = word_to_ix
        self.max_len = max_len
        self.test_dset = test_dset
    
    def __getitem__(self, index):
        data = {}

        row = self.data.iloc[index]
        
        data['tweet'] = row['text']
        data['sentiment'] = row['sentiment']

        if self.test_dset == False:
            data['selected_text'] = row['selected_text']

        sentiment = torch.tensor(self.word_to_ix[row['sentiment']], dtype=torch.long).reshape(1)
        text_tensors = torch.tensor([self.word_to_ix[w] for w in row['text'].lower().split(" ")], dtype=torch.long)
        
        pad_len = self.max_len - sentiment.shape[0] - text_tensors.shape[0]
        if pad_len > 0:
            padding = torch.zeros(pad_len, dtype=torch.long)
            data['inputs'] = torch.cat([sentiment, text_tensors, padding])
        else:
            data['inputs'] = torch.cat([sentiment, text_tensors])

        if self.test_dset == False:
            start_idx, end_idx = self.find_range(row['text'], row['selected_text'])
            data['start_idx'] = torch.tensor(start_idx)
            data['end_idx'] = torch.tensor(end_idx)

        return data

    def __len__(self):
        return len(self.data)

    def find_range(self, str1, str2):
        start_ind = str1.find(str2)
        str1_words = str1.split()
        prev_count = 0
        count = 0
        for i, e in enumerate(str1_words):
            if start_ind >= count and start_ind <= count+len(e):
                return i, i+len(str2.split())-1
            count += len(e)
        return 0, len(str2.split())-1

def get_train_val_loaders(df, test_df, word_to_idx, train_idx, val_idx, batch_size=64):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df, word_to_idx), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df, word_to_idx), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        TweetDataset(test_df, word_to_idx, test_dset=True), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2)

    dataloaders_dict = {"Train": train_loader, "Val": val_loader, "Test": test_loader}

    return dataloaders_dict

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})

    if non_trainable:
        emb_layer.weight.requires_grad = False
 
    return emb_layer, num_embeddings, embedding_dim


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Model is to output
class shallowNN(nn.Module):
    def __init__(self, weights_matrix, input_size, hidden_size, num_layers):
        super(shallowNN, self).__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, non_trainable=True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear1 = nn.Linear(embedding_dim*input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)
        self.linear1.apply(init_weights)
        self.linear2.apply(init_weights)

    def forward(self, x):
        embeds = self.embedding(x)
        # print(embeds.shape)
        embeds = embeds.view(embeds.shape[0], -1)
        # print(embeds.shape)
        out = F.relu(self.linear1(embeds))
        # print(out.shape)
        out = self.linear2(out)
        # print(out.shape)
        # exit()
        start_logits = out[:,0]
        end_logits = out[:,1]
        # start_logits, end_logits = out.split(1, dim=-1)
        # start_logits = start_logits.squeeze(-1)
        # end_logits = start_logits.squeeze(-1)
        return start_logits, end_logits

def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    return ce_loss(start_logits, start_positions) + ce_loss(end_logits, end_positions)


def train(args, model, optimizer, device, dataloaders_dict, epoch, logger):
    for phase in ['Train', 'Val']:
        if phase == 'Train':
            model.train()
        else:
            model.eval()
        
        epoch_loss = 0.0
        jaccard_score = 0.0
        start_avg = 0
        end_avg = 0

        for data in dataloaders_dict[phase]:
            inputs = data['inputs'].to(device)
            start_idx = data['start_idx'].to(device)
            end_idx = data['end_idx'].to(device)
            tweet = data['tweet']

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'Train'):
                start_logits, end_logits = model(inputs)
                # loss = loss_fn(start_logits, end_logits, start_idx, end_idx)
                loss = (torch.abs(start_idx-start_logits) + torch.abs(end_idx-end_logits)).sum()
                if phase == 'Train':
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item() * len(inputs)

                start_idx = start_idx.cpu().detach().numpy()
                end_idx = end_idx.cpu().detach().numpy()
                start_logits = start_logits.cpu().detach().numpy()
                end_logits = end_logits.cpu().detach().numpy()

                # start_avg += start_logits
                # end_avg += end_logits

                # Compute jaccard score
                for i in range(len(inputs)):
                    jaccard_score += compute_jaccard_score(tweet[i], start_idx[i], end_idx[i], start_logits[i], end_logits[i])

        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        jaccard_score = jaccard_score / len(dataloaders_dict[phase].dataset)
        print('Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}'.format(epoch+1, args.epochs, phase, epoch_loss, jaccard_score))
        logger.add_scalar(phase+"/Loss", epoch_loss, epoch+1)
        logger.add_scalar(phase+"/Jaccard", jaccard_score, epoch+1)
        if phase == 'Train':
            # logger.add_scalar(phase+"/start", start_avg / len(dataloaders_dict[phase].dataset), epoch+1)
            # logger.add_scalar(phase+"/end", end_avg / len(dataloaders_dict[phase].dataset), epoch+1)
            for name, param in model.named_parameters():
                logger.add_histogram("Model Params/"+name, param.data, epoch+1)


def predict(args, model, device, dataloaders_dict):
    model.eval()
    
    # jaccard_score = 0.0

    for data in dataloaders_dict['Test']:
        inputs = data['inputs'].to(device)
        tweet = data['tweet']

        with torch.no_grad():
            
            start_logits, end_logits = model(inputs)
            
            start_logits = start_logits.cpu().detach().numpy()
            end_logits = end_logits.cpu().detach().numpy()

            for i in range(len(inputs)):
                # jaccard_score += compute_jaccard_score(tweet[i], start_idx[i], end_idx[i], start_logits[i], end_logits[i])
                print(data['tweet'][i])
                print(start_logits.shape)
                input()
                print((start_logits[i]))
                print((end_logits[i]))
                print(test_get_selected_text(data['tweet'][i], start_logits[i], end_logits[i]))
                input()

    # jaccard_score = jaccard_score / len(dataloaders_dict[phase].dataset)
    # print('Jaccard: {:.4f}'.format(jaccard_score))


def main():

    ### Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    ### Data Preprocessing and Loading

    train_data, test_data = load_data()
    all_data = pd.concat([train_data, test_data])

    if not all(map(os.path.isdir, ['glove6B/glove.6B.50.dat', 'glove6B/glove.6B.50_words.pkl', 'glove6B/glove.6B.50_idx.pkl'])):
        process_glove_vectors()

    glove = load_glove_vectors()
    weights_matrix, word_to_ix = get_embedding(all_data, glove)
    # print(weights_matrix.shape)
    # print(word_to_ix)

    val_num = 500
    indices = np.random.permutation(train_data.shape[0])
    train_inds, val_inds = indices[val_num:], indices[:val_num]
    dataloaders_dict = get_train_val_loaders(train_data, test_data, word_to_ix, train_inds, val_inds, batch_size=args.batch_size)
    
    # X_train, Y_train, X_val, Y_val = process_train_data(train_data, word_to_ix)
    # X_test, Y_test = process_test_data(test_data)

    ### GPU + Pytorch + Tensorboard Logging Setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    hash = hashlib.sha1()
    hash.update(str(time.time()).encode('utf-8'))
    hashname = hash.hexdigest()[:10]
    log_path = "./logs/" + hashname
    logger = SummaryWriter(log_path, flush_secs=0.1)

    ### Model Creation

    model = shallowNN(weights_matrix, 102, 100, 3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    PATH = "model.pt"
    for epoch in range(1, args.epochs + 1):
        train(args, model, optimizer, device, dataloaders_dict, epoch, logger)
        torch.save(model.state_dict(), PATH)
    # example of using the vocab to do look up and passing into embedding
    # print(model.embedding(torch.tensor([word_to_ix["hello"]], dtype=torch.long).to(device)))

    ## Testing Code
    
    # predict(args, model, device, dataloaders_dict)

main()