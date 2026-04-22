import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    pre_residual_blocks = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),   
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )

    return nn.Sequential(
        nn.Residual(pre_residual_blocks),
        nn.ReLU(),
    )


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    model = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes),
    )

    return model


def epoch(dataloader, model, opt=None):
    if opt is not None:
        model.train()
    else:
        model.eval()
    loss_fn = nn.SoftmaxLoss()
    total_loss, total_acc, total_samples = 0.0, 0.0, 0
    
    # 遍历 dataloader
    for i, (x, y) in enumerate(dataloader):
        # 梯度清零
        if opt:
            opt.reset_grad()
            
        out = model(x)
        loss = loss_fn(out, y)
        
        if opt:
            loss.backward()
            opt.step()
        # 使用 .numpy() 或 .item() 获取数值，避免累积计算图
        batch_size = x.shape[0]
        total_samples += batch_size
        total_loss += loss.numpy() * batch_size # 注意要把平均 loss 转回总和
        
        preds = np.argmax(out.numpy(), axis=1)
        total_acc += np.sum(preds == y.numpy())
    return total_loss / total_samples, total_acc / total_samples


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    train_dataset = ndl.data.datasets.MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.datasets.MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz")

    # 将 Dataset 对象传入 DataLoader
    train_loader = ndl.data.DataLoader(dataset=train_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=True)
    test_loader = ndl.data.DataLoader(dataset=test_dataset, 
                                      batch_size=batch_size, 
                                      shuffle=False) 
    
    model = MLPResNet(784, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for e in range(epochs):
        start_time = time.time()
        train_loss, train_acc = epoch(train_loader, model, opt)
        test_loss, test_acc = epoch(test_loader, model)
        print(
            f"Epoch {e+1}: train loss={train_loss:.4f}, train acc={train_acc:.4f}, test loss={test_loss:.4f}, test acc={test_acc:.4f}, time={time.time()-start_time:.2f}s"
        )
    return train_acc, train_loss, test_acc, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
