from model import Generator
from model import Discriminator
from data_loader import neutral_dataset, get_sample
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms as T
import numpy as np
import os
import time
import datetime
from PIL import Image
import glob
import cv2
from collections import OrderedDict
import gc

from face_detection import detection_face_test, detection_and_resize_original, get_face_mesh

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader
        self.sample_loader =  get_sample(config)
        
        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters
        self.test_mode = config.test_mode

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: ", self.device)

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        
        self.rafd_image_dir = config.rafd_image_dir
        
        self.sample_label_dir = config.sample_label_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()
        
        # Add
        if self.dataset == 'RaFD':
            self.crop_size = config.rafd_crop_size
        elif self.dataset == 'CelebA':
            self.crop_size = config.celeba_crop_size
        else :
            self.crop_size = 256
            

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters, mode):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        #self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        #self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        saved_checkpoint_G = torch.load(G_path)
        saved_checkpoint_D = torch.load(D_path)
        
        if self.c_dim == 5:
            self.G.load_state_dict(saved_checkpoint_G, strict = False)
            self.D.load_state_dict(saved_checkpoint_D, strict = False)
       

        ############################################################ new code VER1 USING LINEAR ################################################################################################
        
        # if mode == 'train':

        #     model_G = nn.Linear(8,7).cuda()
        #     new_G = OrderedDict()
        #     G_cnt = 0
        #     for param_tensor in saved_checkpoint_G:
        #         G_cnt += 1
        #         if(G_cnt == 1):
        #             x = saved_checkpoint_G[param_tensor].clone()
                    
        #             x = x.view(x.size(0), x.size(2), x.size(3), -1)
        #             x = model_G(x)
        #             x = x.view(x.size(0), -1, x.size(1), x.size(2))
        #             new_G[param_tensor] = x 
                
        #         else:
        #             new_G[param_tensor] = saved_checkpoint_G[param_tensor]
        #     self.G.load_state_dict(new_G, strict = False)
                    
        #     model_D = nn.Linear(5,4).cuda()
        #     new_D = OrderedDict()
        #     D_cnt = 0
        #     for param_tensor in saved_checkpoint_D:
        #         D_cnt += 1
        #         if(D_cnt == len(saved_checkpoint_D)):
        #             x = saved_checkpoint_D[param_tensor].clone()        
        #             x = x.view(x.size(1), x.size(2), x.size(3), -1)
        #             x = model_D(x)
        #             x = x.view(-1, x.size(0), x.size(1), x.size(2))
        #             new_D[param_tensor] = x 
                
        #         else:
        #             new_D[param_tensor] = saved_checkpoint_D[param_tensor]
        #     self.D.load_state_dict(new_D, strict = False)

        # elif mode == 'test':
        #     self.G.load_state_dict(saved_checkpoint_G, strict = False)
        #     self.D.load_state_dict(saved_checkpoint_D, strict = False)

        ############################################################ new code ################################################################################################ 
        # 
        # 
        ############################################################ new code VER2 DELETE DIM 5 TO 4 #########################################################################
        elif self.c_dim == 4:
            if mode == "train":

                new_G = OrderedDict()
                G_cnt = 0
                for param_tensor in saved_checkpoint_G:
                    G_cnt += 1
                    x = saved_checkpoint_G[param_tensor]
                    if(G_cnt == 1):        
                        img_c1, img_c2, img_c3, black_hair, blonde_hair, brown_hair, male, young = torch.chunk(x, 8, dim=1)
                        y = torch.cat([img_c1, img_c2, img_c3, black_hair, brown_hair, male, young], dim=1)
                        new_G[param_tensor] = y
                    
                    else:
                        new_G[param_tensor] = x
                self.G.load_state_dict(new_G, strict = False)

                new_D = OrderedDict()
                D_cnt = 0
                for param_tensor in saved_checkpoint_D:
                    D_cnt += 1
                    x = saved_checkpoint_D[param_tensor]
                    if(D_cnt == len(saved_checkpoint_D)):        
                        black_hair, blonde_hair, brown_hair, male, young = torch.chunk(x, 5, dim=0)
                        y = torch.cat([black_hair, brown_hair, male, young], dim=0)
                        new_D[param_tensor] = y
                    
                    else:
                        new_D[param_tensor] = x
                self.D.load_state_dict(new_D, strict = False)
            
            elif mode == 'test':
                self.G.load_state_dict(saved_checkpoint_G, strict = False)
                self.D.load_state_dict(saved_checkpoint_D, strict = False)


        ############################################################ new code ################################################################################################   

        
        ############################################################ new code VER2 DELETE DIM 5 TO 6 #########################################################################
        elif self.c_dim == 6:
            if mode == "train":

                new_G = OrderedDict()
                G_cnt = 0
                for param_tensor in saved_checkpoint_G:
                    G_cnt += 1
                    x = saved_checkpoint_G[param_tensor]
                    if(G_cnt == 1):        
                        img_c1, img_c2, img_c3, black_hair, blonde_hair, brown_hair, male, young = torch.chunk(x, 8, dim=1)
                        blended_hair = (black_hair + blonde_hair + brown_hair) / 3
                        y = torch.cat([img_c1, img_c2, img_c3, black_hair, blonde_hair, brown_hair, blended_hair, male, young], dim=1)
                        new_G[param_tensor] = y
                    
                    else:
                        new_G[param_tensor] = x
                self.G.load_state_dict(new_G, strict = False)

                new_D = OrderedDict()   
                D_cnt = 0
                for param_tensor in saved_checkpoint_D:
                    D_cnt += 1
                    x = saved_checkpoint_D[param_tensor]
                    if(D_cnt == len(saved_checkpoint_D)):        
                        black_hair, blonde_hair, brown_hair, male, young = torch.chunk(x, 5, dim=0)
                        blended_hair = (black_hair + blonde_hair + brown_hair) / 3
                        y = torch.cat([black_hair, blonde_hair, brown_hair, blended_hair, male, young], dim=0)
                        new_D[param_tensor] = y
                    
                    else:
                        new_D[param_tensor] = x
                self.D.load_state_dict(new_D, strict = False)

            elif mode == 'test':
                self.G.load_state_dict(saved_checkpoint_G, strict = False)
                self.D.load_state_dict(saved_checkpoint_D, strict = False)
        
        
        
        elif self.c_dim == 7:
            if mode == "train":

                new_G = OrderedDict()
                G_cnt = 0
                for param_tensor in saved_checkpoint_G:
                    G_cnt += 1
                    x = saved_checkpoint_G[param_tensor]
                    if(G_cnt == 1):        
                        img_c1, img_c2, img_c3, black_hair, blonde_hair, brown_hair, male, young = torch.chunk(x, 8, dim=1)
                        blended_hair = (black_hair + blonde_hair + brown_hair) / 3
                        boy = (male + young) / 2
                        y = torch.cat([img_c1, img_c2, img_c3, black_hair, blonde_hair, brown_hair, blended_hair, male, young, boy], dim=1)
                        new_G[param_tensor] = y
                    
                    else:
                        new_G[param_tensor] = x
                self.G.load_state_dict(new_G, strict = False)

                new_D = OrderedDict()   
                D_cnt = 0
                for param_tensor in saved_checkpoint_D:
                    D_cnt += 1
                    x = saved_checkpoint_D[param_tensor]
                    if(D_cnt == len(saved_checkpoint_D)):        
                        black_hair, blonde_hair, brown_hair, male, young = torch.chunk(x, 5, dim=0)
                        blended_hair = (black_hair + blonde_hair + brown_hair) / 3
                        boy = (male + young) / 2
                        y = torch.cat([black_hair, blonde_hair, brown_hair, blended_hair, male, young,boy], dim=0)
                        new_D[param_tensor] = y
                    
                    else:
                        new_D[param_tensor] = x
                self.D.load_state_dict(new_D, strict = False)

            elif mode == 'test':
                self.G.load_state_dict(saved_checkpoint_G, strict = False)
                self.D.load_state_dict(saved_checkpoint_D, strict = False)
        
        ######################################################################## new code #################################################################################### 
    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        # print("lodding logger")
        self.logger = Logger(self.log_dir)
        # print("end")

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)


    def train(self):
        # gc.collect()
        # torch.cuda.empty_cache()
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        
        data_iter = iter(self.sample_loader)
        x_fixed, _ = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        
        expression = glob.glob(self.rafd_image_dir + "/*")
        cnt = 0
        for label in expression:
            if label == "neutral":
                break
            cnt += 1
        c_org = torch.Tensor([])
        for i in range(self.batch_size):
            c_org = torch.cat([c_org, torch.Tensor([cnt])])
        
        
        print("c_org: ", c_org)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
        print("x_fixed.shape: ", x_fixed.shape)
        print("c_org.shape: ", c_org.shape)
        
        
        
        # data_iter = iter(data_loader)
        # x_fixed, c_org = next(data_iter)
        # x_fixed = x_fixed.to(self.device)
        # print("c_org: ", c_org)
        # c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
        # print("x_fixed.shape: ", x_fixed.shape)
        # print("c_org.shape: ", c_org.shape)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters, mode = "train")
            

        # Start training.
        print('Start training...')
        start_time = time.time()
        
        for i in range(start_iters, self.num_iters):
            
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    
                    for c_fixed in c_fixed_list:
                        #print("len(x_fake_list): ", len(x_fake_list))
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    #print("x_concat.shape: ", x_concat.shape)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            #if (i+1) % self.lr_update_step == 0 and (i+1-self.resume_iters) > (self.num_iters - self.num_iters_decay): #바꾼 부분
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):            
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self, img_path):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters, mode = "test")
        
        #Set data loader.
        if self.dataset == 'CelebA': 
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

################################################################## person ######################################################################################
        if self.test_mode == 'person':
            crop_size = self.crop_size
            image_size = self.image_size
            
            img_list = glob.glob('stargan_new_6_leaky/data/test/person/*')
            #img_list = glob.glob('Original/*')
            
            for image in img_list:
                img = Image.open(image)
                name = image.split('/')[-1][0:-8]
                
                transform = []


                transform.append(T.CenterCrop(crop_size))
                transform.append(T.Resize(image_size))
                transform.append(T.ToTensor())
                transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
                transform = T.Compose(transform)

                x_real = transform(img)
                
                x_real = x_real.view(1, 3, image_size, image_size)
                c_org = torch.Tensor([3])

                with torch.no_grad():
                    x_real = x_real.to(self.device)
                    c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                    # Translate images.
                    x_fake_list = [x_real]
                    # x_fake_list = []
                    for c_trg in c_trg_list:
                        x_fake_list.append(self.G(x_real, c_trg))
                        
                    # Save the translated images.
                    # x_concat = torch.cat(x_fake_list, dim=3)
                    # result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(1))
                    # save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)

                    for i, fake in enumerate(x_fake_list):
                        
                        if i == 0:
                            continue
                        if i == 1:
                            result_path = os.path.join(self.result_dir, 'angry', 'angry_{}-images.jpg'.format(name))
                        if i == 2:
                            result_path = os.path.join(self.result_dir, 'fearful', 'fearful_{}-images.jpg'.format(name))
                        if i == 3:
                            result_path = os.path.join(self.result_dir, 'happy', 'happy_{}-images.jpg'.format(name))
                        if i == 4:
                            result_path = os.path.join(self.result_dir, 'neutral', 'neutral_{}-images.jpg'.format(name))
                        if i == 5:
                            result_path = os.path.join(self.result_dir, 'sad', 'sad_{}-images.jpg'.format(name))
                        if i == 6:
                            result_path = os.path.join(self.result_dir, 'surprised', 'surprised_{}-images.jpg'.format(name))

                        save_image(self.denorm(fake.data.cpu()), result_path, nrow=1, padding=0)
                        print('Saved real and fake images into {}...'.format(result_path))
                        
                    
                    x_concat = torch.cat(x_fake_list, dim=3)
                    result_path = os.path.join(self.result_dir, 'total', 'total_{}-images.jpg'.format(name))
                    save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(result_path))
###################################################################################################################################################

        
########################################################################## origin_person #########################################################################
        
        elif self.test_mode == 'origin_person':

            img_list = glob.glob('stargan_new_6_leaky/data/test/origin_person/*')
                
            for path in img_list:
                img_file = cv2.imread(path)
                name = path.split('/')[-1][:-4]
                
                img_list = detection_and_resize_original(img_file)
                
                import matplotlib.pyplot as plt
                
                #img_list[1][0].show()
                #img_list[0]
                totensor = T.ToTensor()
                norm = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                
                img_file = cv2.cvtColor(img_list[0], cv2.COLOR_BGR2RGB)

                original = totensor(img_file)
                original = norm(original)

                img, (x, y, w, h) = img_list[1]
                    
                print(img.size)

                image_size = img.size[0]
                
                transform = []
                
                #transform.append(T.CenterCrop(image_size))
                transform.append(T.ToTensor())
                transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
                transform = T.Compose(transform)
                
                x_real = transform(img)
                tf = T.ToPILImage()
                img_np = tf(x_real)
                x_real = x_real.view(1, 3, image_size, image_size)
                
                c_org = torch.Tensor([3])
                print("Size of x_real is {}".format(x_real.size()))

                with torch.no_grad():
                    x_real = x_real.to(self.device)
                    c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                    # Translate images.
                    x_fake_list = [x_real]
                    x_origin_list = [original]
                    x_mesh_list = [original]

                    for c_trg in c_trg_list:
                        x_fake_list.append(self.G(x_real, c_trg))
                        x_origin_list.append(torch.tensor(original))
                        x_mesh_list.append(torch.tensor(original))


                    for i, fake in enumerate(x_fake_list):
                        
                        if i == 0:
                            continue
                        if i == 1:
                            result_path = os.path.join(self.result_dir, 'angry', 'angry_origin_{}.jpg'.format(name))
                        if i == 2:
                            result_path = os.path.join(self.result_dir, 'fearful', 'fearful_origin_{}.jpg'.format(name))
                        if i == 3:
                            result_path = os.path.join(self.result_dir, 'happy', 'happy_origin_{}.jpg'.format(name))
                        if i == 4:
                            result_path = os.path.join(self.result_dir, 'neutral', 'neutral_origin_{}.jpg'.format(name))
                        if i == 5:
                            result_path = os.path.join(self.result_dir, 'sad', 'sad_origin_{}.jpg'.format(name))
                        if i == 6:
                            result_path = os.path.join(self.result_dir, 'surprised', 'surprised_origin_{}.jpg'.format(name))

                        tranlate_img = fake.data.cpu().squeeze(0)
                        #print("Size of translate_img : {}".format(tranlate_img.shape))
                        
                        from torchvision.transforms.functional import to_pil_image

                        for j in range(3):
                            for k in range(y, y+h):
                                x_origin_list[i][j][k][x:x+w] = tranlate_img[j][k-y]
                        
                        face_dict, mesh_img = get_face_mesh(to_pil_image(0.5 *x_origin_list[i] +0.5))
                        
                        tf= T.Compose([T.ToTensor(), T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
                        mesh_tensor = tf(mesh_img)
                        
                        for j in range(3):
                            for y1, x_list in face_dict.items():
                                if len(x_list) == 1:
                                    continue
                                x1, x2 = x_list
                                x_mesh_list[i][j][y1][x1:x2] = mesh_tensor[j][y1][x1:x2]
                        
                        save_image(self.denorm(x_mesh_list[i].data.cpu()), result_path, nrow=1, padding=0)
                        print('Saved real and fake images into {}...'.format(result_path))
                        
                        
                    x_concat = torch.cat(x_mesh_list, dim=2)
                    result_path = os.path.join(self.result_dir, 'mesh_result', 'total_origin_{}.jpg'.format(name))
                    save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(result_path))
    ####################################################################################################################################################
        
        ################################################### Until this part. #######################################################

            # for i, (x_real, c_org) in enumerate(data_loader):
            #     print("x_real: ", x_real.size())
            #     print(c_org)
            #     # Prepare input images and target domain labels.
            #     x_real = x_real.to(self.device)
            #     c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

            #     # Translate images.
            #     x_fake_list = [x_real]
            #     for c_trg in c_trg_list:
            #         x_fake_list.append(self.G(x_real, c_trg))

            #     # Save the translated images.
            #     x_concat = torch.cat(x_fake_list, dim=3)
            #     result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
            #     save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            #     print('Saved real and fake images into {}...'.format(result_path))

    # ###########################################바꾼버전#####################################################################
    # def test(self):
    # #"""Translate images using StarGAN trained on a single dataset."""
    # # Load the trained generator.
    #     self.restore_model(self.test_iters)
        
    #     # Set data loader.
    #     # if self.dataset == 'CelebA':
    #     #     data_loader = self.celeba_loader
    #     # elif self.dataset == 'RaFD':
    #     #     data_loader = self.rafd_loader

    #     ####################################################

    #     crop_size = 256
    #     image_size = 128

    #     img_file = glob.glob('data/KDEF_Noside/test/neutral/*')
    #     #img = Image.open('data/KDEF_Noside/test/neutral/resize_jimin.jpg')
    #     idx = 0
    #     for file in img_file:
    #         img = Image.open(file)
    #         transform = []
    #         transform.append(T.CenterCrop(crop_size))
    #         transform.append(T.Resize(image_size))
    #         transform.append(T.ToTensor())
    #         transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    #         transform = T.Compose(transform)

    #         x_real = transform(img)
    #         x_real = x_real.view(1, 3, 128, 128)
    #         c_org = torch.Tensor([3])

    #         with torch.no_grad():
    #             x_real = x_real.to(self.device)
    #             c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

    #             # Translate images.
    #             x_fake_list = [x_real]
    #             # x_fake_list = []
    #             for c_trg in c_trg_list:
    #                 x_fake_list.append(self.G(x_real, c_trg))
    #                 print(c_trg)
    #             # Save the translated images.
    #             x_concat = torch.cat(x_fake_list, dim=3)
    #             str = "G_" + file.split("/")[4]
    #             #print(str)
    #             result_path = os.path.join(self.result_dir, str)
    #             save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
    #             idx += 1
                
    # #######################################################################################################################
