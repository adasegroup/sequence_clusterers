import torch
import torch.optim as optim
from argparse import ArgumentParser
import torch.nn.functional as F
from preprocessing import *
from model_cnn import *
from torch.utils.data import Dataset, DataLoader
import json
#training model
class Run:
    @staticmethod
    def train(model, data, params):
        
           
        f = pd.read_csv(f"Amazon_short_text/clusters.csv")
        y = f["category"].tolist()
        
        data = torch.LongTensor(data)
        train = data, y
        loader_train = DataLoader(train, batch_size=params.batch_size)
        y= torch.FloatTensor(y)
        # Define optimizer
        optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate)

        # Starts training phase
        for epoch in range(params.epochs):
        # Set model in training model
            model.train()
            predictions = []
            # Starts batch training 

            # Feed the model
            y_pred = model(data)

            # Loss calculation
            loss = F.binary_cross_entropy(y_pred, y)

            # Clean gradientes
            optimizer.zero_grad()

            # Gradients calculation
            loss.backward()

            # Gradients update
            optimizer.step()
            return model.union

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path_to_data", type=str, default= "Amazon_short_text")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--embedding_size", type=int, default=16)
    parser.add_argument("--out_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=10000)
    parser.add_argument("--num_words", type=float, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_to", type=str, default="Amazon_text.json")
    configs = parser.parse_args()
    data = prepare_data(configs.num_words, configs.seq_len, configs.path_to_data )
    # Initialize the model
    model = TextClassifier(configs)
    
    # Training - Evaluation  to get embeddings
    union = Run().train(model, data, configs)
    print(union.size()[0])
    emb = {i: str(list(union[i, :].detach().numpy())) for i in range(union.size()[0])}
    with open(configs.save_to, 'w') as f:
        json.dump(emb, f)

    
    