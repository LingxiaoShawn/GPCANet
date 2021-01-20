import torch
def train(net, optimizer, criterion, data, device, minibatch):
    if minibatch:
        return train_minibatch(net, optimizer, criterion, data, device)
    else:
        return train_fullbatch(net, optimizer, criterion, data)
    
def evaluate(net, criterion, data, device, minibatch):
    # modify later, currently we only evaluate in full batch mode
    if minibatch:
        data.to(device)
        results = evaluate_fullbatch(net, criterion, data)
        data.to('cpu')
    else:
        results = evaluate_fullbatch(net, criterion, data)
    return results
        
def train_fullbatch(net, optimizer, criterion, data):
    net.train()
    optimizer.zero_grad()
    output = net(data, minibatch=False) # for GPCA training
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    acc = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

def train_minibatch(net, optimizer, criterion, dataloader, device):
    net.train()
    total_loss = total_examples = total_correct = 0
    for data in dataloader:
        data = data.to(device)
        if data.train_mask.sum() == 0:
            continue
        optimizer.zero_grad()
        output = net(data, minibatch=True)
#         print(output.shape, data.train_mask.shape)
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        num_examples = data.train_mask.sum().item()
        total_loss += loss.item() * num_examples
        total_correct += accuracy(output[data.train_mask], data.y[data.train_mask]).item() * num_examples
        total_examples += num_examples
        
    return total_loss / total_examples, total_correct / total_examples

def evaluate_fullbatch(net, criterion, data):
    with torch.no_grad():
        net.eval()
        output = net(data, minibatch=False)
        train_acc = accuracy(output[data.train_mask], data.y[data.train_mask])
        val_acc = accuracy(output[data.valid_mask], data.y[data.valid_mask])
        test_acc = accuracy(output[data.test_mask], data.y[data.test_mask])
    return train_acc.item(), val_acc.item(), test_acc.item()

def evaluate_minibatch(net, criterion, dataloader, device):

    pass
    
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)