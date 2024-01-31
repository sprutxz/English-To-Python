from model import *

# Training the model
NUM_EPOCHS = 10

for epoch in range(1,NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer)
    end_time = timer()
    val_loss = evaluate(model)
    print(f'Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}')
    print(f'Epoch time: {(end_time - start_time):.3f}s')
    
torch.save(model.state_dict(), 'model.pth')