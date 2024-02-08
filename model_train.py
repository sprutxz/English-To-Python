from model import *
import os

# Training the model
NUM_EPOCHS = 200
train_loss_data = []
val_loss_data = []
train_acc_data = []
val_acc_data = []    

if os.path.exists('model.pth'):
    model.load_state_dict(torch.load('model.pth'))
print(sum(p.numel() for p in model.parameters() if p.requires_grad))    
for epoch in range(NUM_EPOCHS):
    start_time = timer()
    train_loss,train_acc = train_epoch(model, optimizer)
    train_loss_data.append(train_loss)
    train_acc_data.append(train_acc)
    end_time = timer()
    val_loss,val_acc = evaluate(model)
    val_loss_data.append(val_loss)
    val_acc_data.append(val_acc)
    print(f'Epoch: {epoch+1}\n Train loss: {train_loss:.3f}, Train Acc: {train_acc}\n Val loss: {val_loss:.3f}, Val Acc: {val_acc}', flush=True)
    print(f'Epoch time: {(end_time - start_time):.3f}s', flush=True)
    
torch.save(model.state_dict(), 'model.pth')
plt.plot(range(1, NUM_EPOCHS+1), train_loss_data, label='Train Loss')
plt.plot(range(1, NUM_EPOCHS+1), val_loss_data, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show(block=True)
plt.savefig('loss.png')