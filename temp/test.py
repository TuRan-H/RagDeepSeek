import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

# 设置随机种子，确保结果可复现
torch.manual_seed(42)

# 参数设置
batch_size = 128
epochs = 5
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载MNIST数据集
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义Transformer模型的构建块

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections and split into heads
        q = self.wq(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Create causal mask (lower triangular)
        if mask is None:
            seq_len = x.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Apply mask
        scores = scores.masked_fill(mask, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights
        context = torch.matmul(attn_weights, v)
        
        # Concatenate heads and apply final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.wo(context)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)  # GELU activation as used in GPT-2
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention block with residual connection and layer norm
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward block with residual connection and layer norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, input_size=784, d_model=256, num_heads=8, num_layers=12, d_ff=1024, dropout=0.2, num_classes=10):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layer to convert flattened MNIST images to embeddings
        self.embedding = nn.Linear(28, d_model)  # 每行28像素
        
        # Positional encoding (simplified for MNIST since we don't have sequence)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 28, d_model))  # 28 positions for 28x28 image rows
        
        # Decoder layers - 修改为12层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final Layer Norm
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.fc_out = nn.Linear(d_model, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape image to [batch_size, 28, 28]
        x = x.view(batch_size, 28, 28)
        
        # Embed each row of the image
        x = self.embedding(x)  # [batch_size, 28, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply transformer decoder layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply final layer norm
        x = self.norm(x)
        
        # Use the representation of the "last token" for classification
        x = x[:, -1, :]
        
        # Project to output classes
        x = self.fc_out(x)
        
        return x

# 实例化模型并移至设备
model = TransformerDecoder().to(device)
print(f"使用设备: {device}")
print(model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        # 更新进度条信息
        pbar.set_postfix({'loss': f'{loss.item():.6f}', 
                          'acc': f'{100. * correct / ((batch_idx+1) * len(data)):.2f}%'})
    
    train_loss /= len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'训练集: 平均损失: {train_loss:.4f}, 准确率: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)')
    return train_loss, accuracy

# 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='[Test]')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # 更新进度条信息
            pbar.set_postfix({'loss': f'{test_loss / (pbar.n + 1):.6f}', 
                             'acc': f'{100. * correct / ((pbar.n + 1) * len(data)):.2f}%'})
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'测试集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return test_loss, accuracy

# 训练和评估模型
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in tqdm(range(1, epochs + 1), desc='训练进度'):
    train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
    test_loss, test_acc = test(model, device, test_loader)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

# 确保results目录存在
import os
os.makedirs('./results', exist_ok=True)

# 保存模型
torch.save(model.state_dict(), './results/mnist_mlp.pth')
print('模型已保存至 ./results/mnist_mlp.pth')

# 打印最终结果
print(f'最终测试准确率: {test_accuracies[-1]:.2f}%')
