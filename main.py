import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Current device: {device}')


class HandwritingModel(torch.nn.Module):
    def __init__(self):
        super(HandwritingModel, self).__init__()
        self.fc_layer = torch.nn.Linear(784, 10)

    def forward(self, input_tensor):
        input_tensor = input_tensor.view(-1, 784)
        input_tensor = self.fc_layer(input_tensor)
        return input_tensor


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)

handwriting_model = HandwritingModel()
handwriting_model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(handwriting_model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = handwriting_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print("The training process completed.")