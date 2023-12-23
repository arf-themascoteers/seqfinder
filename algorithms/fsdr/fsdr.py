import math
from sklearn.metrics import mean_squared_error, r2_score
from approximator import get_splines
import torch
from algorithms.fsdr.ann import ANN
from datetime import datetime
import os
import my_utils


class FSDR:
    def __init__(self, rows, original_feature_size, target_feature_size, seq=False, mode="linear"):
        #mode = linear, fc, skip
        self.seq = seq
        self.original_feature_size = original_feature_size
        self.target_feature_size = target_feature_size
        self.model = ANN(self.target_feature_size, self.original_feature_size, seq, mode)
        self.lr = 0.001
        self.weight_decay = self.lr/10
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.epochs = my_utils.get_epoch(rows, self.target_feature_size)
        self.csv_file = os.path.join("results", f"fsdr-{seq}-{target_feature_size}-{str(datetime.now().timestamp()).replace('.', '')}.csv")
        self.start_time = datetime.now()
        print("Learnable Params",sum(p.numel() for p in self.model.parameters() if p.requires_grad))


    def get_elapsed_time(self):
        return round((datetime.now() - self.start_time).total_seconds(),2)

    def fit(self, X, y, X_validation, y_validation):
        print(f"X,X_validation: {X.shape} {X_validation.shape}")
        self.write_columns()
        self.model.train()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        spline = get_splines(X, self.device)
        X_validation = torch.tensor(X_validation, dtype=torch.float32).to(self.device)
        spline_validation = get_splines(X_validation, self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        y_validation = torch.tensor(y_validation, dtype=torch.float32).to(self.device)
        for epoch in range(self.epochs):
            y_hat = self.model(spline)
            loss_1 = self.criterion(y_hat, y)
            loss_2 = 0
            loss = loss_1 + loss_2
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            row = self.dump_row(epoch, X, spline, y, X_validation, spline_validation, y_validation)
            if epoch%50 == 0:
                print("".join([str(i).ljust(20) for i in row]))
        return self.get_indices()

    def evaluate(self,X, spline,y):
        self.model.eval()
        y_hat = self.model(spline)
        y_hat = y_hat.reshape(-1)
        y_hat = y_hat.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        r2 = r2_score(y, y_hat)
        rmse = math.sqrt(mean_squared_error(y, y_hat))
        self.model.train()
        return max(r2,0), rmse

    def write_columns(self):
        columns = ["epoch","train_r2","validation_r2","train_rmse","validation_rmse","time"]
        for index,p in enumerate(self.model.get_indices()):
            columns.append(f"band_{index+1}")
        print("".join([c.ljust(20) for c in columns]))
        with open(self.csv_file, 'w') as file:
            file.write(",".join(columns))
            file.write("\n")

    def dump_row(self, epoch, X, spline, y, X_validation, spline_validation, y_validation):
        train_r2, train_rmse = self.evaluate(X, spline, y)
        test_r2, test_rmse = self.evaluate(X_validation, spline_validation, y_validation)
        row = [train_r2, test_r2, train_rmse, test_rmse]
        row = [round(r,5) for r in row]
        row = [epoch] + row + [self.get_elapsed_time()]
        for p in self.model.get_indices():
            row.append(self.indexify_raw_index(p))
        with open(self.csv_file, 'a') as file:
            file.write(",".join([f"{x}" for x in row]))
            file.write("\n")
        return row

    def indexify_raw_index(self, raw_index):
        raw_index = torch.mean(raw_index, dim=0)
        multiplier = self.original_feature_size
        return round(raw_index.item() * multiplier)

    def get_indices(self):
        indices = sorted([self.indexify_raw_index(p) for p in self.model.get_indices()])
        return list(dict.fromkeys(indices))

    def transform(self, x):
        return x[:,self.get_indices()]

    def predict_it(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        spline = get_splines(X, self.device)
        y_hat = self.model(spline)
        return y_hat.detach().cpu().numpy()
