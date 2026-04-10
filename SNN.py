import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torch.nn.functional as F

# 檢查 CUDA 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")
if torch.cuda.is_available():
    print(f"GPU 型號: {torch.cuda.get_device_name(0)}")


# SNN 模型定義
class SpikingNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10,
                 beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25)):
        super(SpikingNN, self).__init__()

        # 網路層定義
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc3 = nn.Linear(hidden_size, output_size)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, num_steps=25):
        # 初始化隱藏狀態
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # 記錄輸出脈衝
        spk3_rec = []
        mem3_rec = []

        # 時間步長迭代
        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec), torch.stack(mem3_rec)


# 訓練函數
def train_snn(model, train_loader, optimizer, num_epochs=10, num_steps=25):
    model.train()
    loss_fn = SF.ce_rate_loss()  # 使用脈衝計數的交叉熵損失

    for epoch in range(num_epochs):
        train_loss = 0
        correct = 0
        total = 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            data = data.view(data.size(0), -1)  # 展平輸入

            optimizer.zero_grad()

            # 前向傳播
            spk_rec, mem_rec = model(data, num_steps)

            # 計算損失
            loss = loss_fn(spk_rec, targets)

            # 反向傳播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 計算準確率（使用最後時刻的膜電位）
            _, predicted = mem_rec[-1].max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        avg_loss = train_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


# 測試函數
def test_snn(model, test_loader, num_steps=25):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            data = data.view(data.size(0), -1)

            spk_rec, mem_rec = model(data, num_steps)

            _, predicted = mem_rec[-1].max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f'測試準確率: {accuracy:.2f}%')
    return accuracy


# 主程式範例
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # 數據載入（以 MNIST 為例）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # 建立模型
    model = SpikingNN(input_size=784, hidden_size=256, output_size=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("開始訓練...")
    train_snn(model, train_loader, optimizer, num_epochs=5, num_steps=25)

    print("\n開始測試...")
    test_snn(model, test_loader, num_steps=25)

    # 儲存模型
    torch.save(model.state_dict(), 'snn_model.pth')
    print("模型已儲存至 snn_model.pth")