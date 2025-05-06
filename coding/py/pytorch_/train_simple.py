from pathlib import Path
from datetime import datetime
import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from .dataset import get_loader
from .network import NeuralNetwork


# DEBUG
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.autograd.set_detect_anomaly(True)
#choose most eff. alg. for convolution
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)


# OPTIONS
DS_DIR = Path('/nas/people/user/ds/animals')
EXP_DIR = Path('/nas/people/user/experiments/gans/wgan_gp_animals')

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BATCH_SIZE = 64
IMG_SIZE = 128
LR = 1e-3
EPOCHS = 5
DISPLAY_STEP = 1

#################### LOGS ####################
writer = SummaryWriter(str(EXP_DIR))

#checkpoints_dir = EXP_DIR / datetime.now().strftime('%Y-%m-%d_%H%M')
checkpoints_dir = EXP_DIR / 'checkpoints'
checkpoints_dir.mkdir(parents=True, exist_ok=True)

log_path = EXP_DIR / 'loss_log.txt'
with open(str(log_path), "w") as log_file:
    now = time.strftime("%c")
    log_file.write('================ Training Loss (%s) ================\n' % now)


# DATA & MODEL
paths = list(DS_DIR.glob('*/*/*.jpg'))
train_dl = get_loader(IMG_SIZE, BATCH_SIZE, paths, train=True)
val_dl = get_loader(IMG_SIZE, BATCH_SIZE, paths, train=False)

# sample = next(iter(dataloader))

model = NeuralNetwork().to(DEVICE)


# TRAIN
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# Load 1
model.load_state_dict(torch.load('model_weights.pth'))
# Load 2
model = torch.load('model.pth')
# Load 3
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start = checkpoint['epoch']+1
loss = checkpoint['loss']

train_losses = []
val_losses = []
for epoch in tqdm(range(EPOCHS)):
    print(f'Epoch {epoch+1}')

    ### Train
    model.train()
    for batch, x in enumerate(train_dl):
        x = x.to(DEVICE)
        # Backpropagation
        w = torch.randn(5, 3, requires_grad=True)
        w.requires_grad_(True)  # for existing variable 

        logits = model(x)
        pred_probab = nn.Softmax(dim=1)(logits)     
        y_pred = pred_probab.argmax(1)
        print(f"Predicted class: {y_pred}")      

        loss_train = loss_fn(logits, torch.zeros(3))  # Note: nn.CrossEntropyLoss has SoftMax built in!!
        print('Gradient function for prediction =', logits.grad_fn)
        
        optimizer.zero_grad()
        '''
        !!!!!
        We can only perform gradient calculations using `backward` once on a given graph, for performance reasons. 
        If we need to do several `backward` calls on the same graph, we need to pass `retain_graph=True` to the backward call.
        '''
        loss_train.backward()
        optimizer.step()

        train_losses += [loss_train.item()]

        if batch % 100 == 0:
            loss_train, current = loss_train.item(), batch * len(X)
            print(f"loss: {loss_train:>7f}  [{current:>5d}/{len(train_dl.dataset):>5d}]")


        ### Save
        if epoch%2 == 0:
            torch.save(model.state_dict(), 'model_weights.pth') # Weights
            torch.save(model, 'model.pth')                      # Whole model

            # General checkpoint 
            loss, current = loss_train.item()
            torch.save(
                {
                    'epoch': epoch, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, 
                'model.pt')

        ### Validate
        model.eval()

        size = len(train_dl.dataset)
        num_batches = len(train_dl)
        val_loss, correct = 0, 0

        # when the model is used for inference we don't need to track computations
        # --> use tensor.detach() or
        # --> surround the computation code with `torch.no_grad()` block:
        with torch.no_grad():
            for X, y in val_dl:
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        val_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
        
        val_losses += [val_loss]
       

    # end of an epoch
    running_train_loss = sum(train_losses)/len(train_losses)
    running__val_loss = sum(val_losses)/len(val_losses)
    writer.add_scalar('Loss/training', running_train_loss, epoch)
    writer.add_scalar('Loss/validation', running__val_loss, epoch)


    def plot_images(batch, size, nrow=4, path=None):
        figure = plt.figure(figsize=(8,8))
        plt.axis("off")
        image_unflat = batch.detach().cpu().view(-1, *size)
        num_images=nrow*nrow
        image_grid = make_grid(image_unflat[:num_images], nrow=nrow, normalize=True)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        if path: plt.savefig(str(path))
        return figure

    noise = torch.randn(16, 100) 
    fake = model(noise)    
    figure = plot_images(fake, (3,IMG_SIZE,IMG_SIZE))
    writer.add_figure('fakes', figure, epoch)

    if (epoch+1) % DISPLAY_STEP == 0:        
        images_path = EXP_DIR / f'epoch{epoch+1}.png'
        plot_images(fake, (3,IMG_SIZE,IMG_SIZE), path=images_path)


def save_loss_plot(train_losses, val_losses, path):
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Train loss', 'Validation losses'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig(str(path))

save_loss_plot(train_losses, val_losses, EXP_DIR / 'losses.png')

writer.flush()
writer.close()