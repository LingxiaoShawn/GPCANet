def train(net, optimizer, criterion, data):
    net.train()
    optimizer.zero_grad()
    output = net(data, use_cache=True) # for GPCA training
    loss = criterion(output[data.train_idx], data.y[data.train_idx])
    acc = accuracy(output[data.train_idx], data.y[data.train_idx])
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

def evalulate(net, criterion, data):
    net.eval()
    output = net(data, use_cache=True)
    train_acc = accuracy(output[data.train_idx], data.y[data.train_idx])
    val_acc = accuracy(output[data.valid_idx], data.y[data.valid_idx])
    test_acc = accuracy(output[data.test_idx], data.y[data.test_idx])
    return train_acc.item(), val_acc.item(), test_acc.item()

# def val(net, criterion, data):
#     net.eval()
#     output = net(data)
#     loss_val = criterion(output[data.valid_idx], data.y[data.valid_idx])
#     acc_val = accuracy(output[data.valid_idx], data.y[data.valid_idx])
#     return loss_val, acc_val

# def test(net, criterion, data):
#     net.eval()
#     output = net(data)
#     loss_test = criterion(output[data.test_idx], data.y[data.test_idx])
#     acc_test = accuracy(output[data.test_idx], data.y[data.test_idx])
#     return loss_test, acc_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)