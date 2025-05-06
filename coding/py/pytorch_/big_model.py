import torch

from .architecture.generator import Generator
from .architecture.discriminator import Discriminator
from loss import Losses
from models.flow_model import FlowModel




class MainModel():
    def __init__(self, num_classes, device, train=False, lr=None):
        self.device = device
        self.generator = Generator(num_classes).to(device)

        if train:
            self.discriminator = Discriminator(3+1).to(device)
            self.flow_model = FlowModel(device)

            self.optimizer_main = torch.optim.Adam(list(self.generator.parameters()), lr=lr)
            self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr)       
        
            self.losses = Losses(num_classes, device)

            self.persistence = ['generator', 'discriminator', 'optimizer_main', 'optimizer_disc']


    def train_step(self, batch):
        rgbs = batch['rgb']
        gt_depths = batch['depth']
        gt_seg = batch['seg']

        prev_depth = gt_depths[0].to(self.device)

        for i in range(1, len(rgbs)):
            if i == 0:
                continue

            prev_rgb = rgbs[i-1].to(self.device)
            curr_rgb = rgbs[i].to(self.device)

            prev_depth_gt = gt_depths[i-1].to(self.device)
            curr_depth_gt = gt_depths[i].to(self.device)

            curr_seg_gt = gt_seg[i].to(self.device)

            # GENRATOR
            depth_fake_pyramid, curr_seg = self.generator(curr_rgb, prev_rgb, prev_depth)
            curr_depth, _, _, _ = depth_fake_pyramid

            disc_out = self.discriminator(torch.cat([curr_rgb, curr_depth], 1))
            
            flow_loss = self.flow_model.calculate_loss(((curr_depth_gt)), ((prev_depth_gt)), ((curr_depth)), ((prev_depth)))
            self.losses.calculate_depth_loss(depth_fake_pyramid, curr_depth_gt, curr_rgb, disc_out, flow_loss)
            self.losses.calculate_segmentation_loss(curr_seg, curr_seg_gt)     

            self.optimizer_main.zero_grad()
            self.losses.depth_loss.backward(retain_graph=True)
            self.losses.segmentation_loss.backward(retain_graph=True)
            ''' 
            Note: if we have a graph like this: 
                              --> d
            a --> b --> c --<
                              --> e

            when we do d.backward(), that is fine. 
            After this computation, the parts of the graph that calculate d will be freed by default to save memory. 
            So if we do e.backward(), the error message will pop up. In order to do e.backward(), 
            we have to set the parameter retain_graph to True in d.backward()
            '''

            # DISCRIMINATOR
            self.optimizer_disc.zero_grad()
            disc_out_real = self.discriminator(torch.cat([curr_rgb, curr_depth_gt], 1))
            disc_out_fake = self.discriminator(torch.cat([curr_rgb, curr_depth.detach()], 1))
            disc_loss = self.losses.calculate_disc_loss(disc_out_real, disc_out_fake.detach())
            disc_loss.backward()

            # assign
            prev_depth = curr_depth
        
        self.optimizer_disc.step()
        self.optimizer_main.step()
        return self.losses.get_losses()


    def test_sequence(self, rgbs, depth_init):

        depths = []
        depths.append(depth_init.to(self.device))
        
        for i in range(1, len(rgbs)):
            rgb_prev = rgbs[i-1].to(self.device)
            rgb_curr = rgbs[i].to(self.device)

            depth_prev = depths[i-1].to(self.device)  

            depth_pyramid, _ = self.generator(rgb_curr, rgb_prev, depth_prev)
            depth_curr, _, _, _ = depth_pyramid
            depths.append(depth_curr)

        return depths


    def save_networks(self, save_path):
        dict2save = {}
        for part_name in self.persistence: 
            model_part = getattr(self, part_name)
            dict2save[f'state_dict_{part_name}'] = model_part.state_dict()

        torch.save(dict2save, str(save_path))


    def load_networks(self, load_path):
        state_dict = torch.load(load_path, map_location=str(self.device))
  
        for part_name in self.persistence: 
            part = getattr(self, part_name, 'not found')
            if part != 'not found':
                part.load_state_dict(state_dict[f'state_dict_{part_name}'])