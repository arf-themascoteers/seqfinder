import torch.nn as nn
import torch
import my_utils
from sklearn.metrics import r2_score


class ModelANN(nn.Module):
    def __init__(self, X):
        super().__init__()
        torch.manual_seed(3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rows = X.shape[0]
        features = X.shape[1]
        self.linear = my_utils.get_linear(rows, features)
        print(self.linear)
        self.epoch = my_utils.get_epoch(rows, features)
        self.lr = my_utils.get_lr(rows, features)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.to(self.device)

    def forward(self, X):
        return self.linear(X.to(self.device)).reshape(-1)

    def create_optimizer(self):
        weight_decay = self.lr/10
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)

    def fit(self, X, y):
        self.train()
        optimizer = self.create_optimizer()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        for epoch in range(self.epoch):
            y_hat = self(X)
            loss = self.criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch%50 == 0:
                print(f"{epoch}: {round(loss.item(),5)} "
                      f"{round(r2_score(y.detach().cpu().numpy(), self.predict(X.detach().cpu().numpy(), False)),5)}")

    def predict(self, X, temp=False):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = self(X)
        if temp:
            self.train()
        return y.detach().cpu().numpy()

