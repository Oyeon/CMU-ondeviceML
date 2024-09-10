import torch

def train_one_epoch(epoch_index, model, train_loader, optimizer, loss_fn, tb_writer):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(f'  batch {i + 1} loss: {last_loss}')
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.0
    return running_loss / len(train_loader)

def validate(model, val_loader, loss_fn):
    model.eval()
    running_vloss = 0.0
    running_vcorrects = 0
    total_vsamples = 0
    with torch.no_grad():
        for vinputs, vlabels in val_loader:
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()
            _, vpreds = torch.max(voutputs, 1)
            running_vcorrects += torch.sum(vpreds == vlabels).item()
            total_vsamples += vlabels.size(0)
    avg_vloss = running_vloss / len(val_loader)
    validation_accuracy = running_vcorrects / total_vsamples
    return avg_vloss, validation_accuracy