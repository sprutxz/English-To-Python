from model import *

# Training the model
NUM_EPOCHS = 50
model.load_state_dict(torch.load('model.pth'))
for epoch in range(1,NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer)
    end_time = timer()
    val_loss = evaluate(model)
    print(f'Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}', flush=True)
    print(f'Epoch time: {(end_time - start_time):.3f}s', flush=True)
    
torch.save(model.state_dict(), 'model.pth')
