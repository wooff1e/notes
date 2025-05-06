from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter

from data.dataloader import get_loader
from models.main_model import MainModel
from logger import Logger
from timer import Timer
from python_core.paths import get_paths



def save_depths(depths, save_dir):
    for j, depth in enumerate(depths):
        depth = depth.detach().cpu().numpy().squeeze()
        depth = (depth*255).astype(np.uint8)
        depth = Image.fromarray(depth)
        save_path = save_dir / f'{j:05d}.png'
        depth.save(save_path)


# DEBUG
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.autograd.set_detect_anomaly(True)
#choose most eff. alg. for convolution
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)


# OPTIONS
EXP_NAME = 'debug'
DS_NAME = 'SYNTHIA-AL_2019'
OVERFIT = False
LOAD_LAST_CHKPT = False

FRAMES_PER_SAMPLE = 4
CLASSES = 26
BATCH_SIZE = 2
LR = 0.001
EPOCHS = 20
LOG_STEP = 10
SAVE_STEP = 10


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

exp_dir, ds_dir = get_paths()
exp_dir = exp_dir / EXP_NAME
ds_dir = ds_dir / DS_NAME


# LOGS
writer = SummaryWriter(str(exp_dir))

checkpoints_dir = exp_dir / 'checkpoints'
checkpoints_dir.mkdir(parents=True, exist_ok=True)

logger = Logger(exp_dir / 'logs.txt')
if OVERFIT: logger.log(f'OVERFIT')
logger.log(f'frames per sample: {FRAMES_PER_SAMPLE}')
logger.log(f'batch size: {BATCH_SIZE}')


# DATA
test_dl = get_loader(ds_dir / 'test', FRAMES_PER_SAMPLE, 1, train=False)
test_sample = next(iter(test_dl))
sample_rgbs = test_sample['rgb']
depth_init = test_sample['depth'][0]

if OVERFIT: 
    train_dl = [test_sample,]
else:
    train_dl = get_loader(ds_dir / 'train', FRAMES_PER_SAMPLE, BATCH_SIZE, train=True)


print('data loaded')

# MODEL
print('creating model...')
model = MainModel(CLASSES, device, train=True, lr=LR)

if LOAD_LAST_CHKPT:
    checkpoints = sorted(list(checkpoints_dir.glob('*.pth')))
    load_path = str(checkpoints[-1])
    print(f'using checkpoint: {load_path}')
    model.load_networks(load_path)

# TRAIN
epoch_timer = Timer('epoch')
step_timer = Timer('step')

for epoch in range(1, EPOCHS+1):
    print('\n\nEPOCH', epoch, '/', EPOCHS)

    for i, data in enumerate(train_dl):
        print(f'batch {i}/{len(train_dl)}')
        global_step = (epoch-1) * len(train_dl) + i

        losses = model.train_step(data)
        
        writer.add_scalar('Loss/depth_loss', losses['depth_loss'], global_step)
        writer.add_scalar('Loss/segmentation_loss', losses['segmentation_loss'], global_step)

        if (i+1) % LOG_STEP == 0:
            logger.log_losses(losses)   
            avg = step_timer.running_avg         
            logger.log(f'average time per step: {avg/60.0:.2f} min ({avg:.2f} sec)')

        if (i+1) % SAVE_STEP == 0:
            model.save_networks(checkpoints_dir / f'checkpoint_ep{epoch}_st{i}.pth')
            sequence = test_sample['rgb']
            save_dir = exp_dir / f'images_ep{epoch}_st{i}'
            save_dir.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                depths = model.test_sequence(sample_rgbs, depth_init)
                save_depths(depths, save_dir)

        step_timer.reset()
    epoch_timer.reset()

avg = epoch_timer.running_avg
logger.log(f'average time per epoch: {avg/60.0:.2f} min ({avg:.2f} sec)')

writer.flush()
writer.close()