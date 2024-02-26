# 导入所需的库
import random
import time

import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

# 假设X是特征矩阵，y是对应的标签
# 请替换下面的X和y为你的实际数据
# X的每一行是一个信号的特征，y的每个元素是对应信号的类别标签
# 你可能需要根据你的数据格式进行适当的修改
import numpy as np

# 设置随机种子，以便结果可复现
from utils import load_data
class ECGDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.x)

np.random.seed(42)

config = {
        'seed': 42,  # the random seed
        'test_ratio': 0.3,  # the ratio of the test set
        'num_epochs': 30,
        'batch_size': 128,
        'lr': 0.001,
    }
# 设置随机种子
seed_value = config['seed']
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

# X_train,y_train is the training set
# X_test,y_test is the test set
X_train, X_test, y_train, y_test = load_data(config['test_ratio'], config['seed'])
train_dataset, test_dataset = ECGDataset(X_train, y_train), ECGDataset(X_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=config['seed'])

# 在训练集上训练模型
rf_classifier.fit(X_train, y_train)

start_time = time.time()
# 在测试集上进行预测
y_pred = rf_classifier.predict(X_test)
end_time = time.time()
inference_time=end_time-start_time

# 计算准确度
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确度: {accuracy}")


print(f"平均运行时间: {inference_time:.4f} 秒")