import torch
from torch.autograd import Variable
from .base_model import BaseModel
from . import network
from util import task, util
import ntpath,os


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def __init__(self, opt):
        assert (not opt.isTrain)
        BaseModel.__init__(self, opt)

        self.visual_names = ['img_truth', 'img_mask', 'img_fake', 'offset']
        self.value_names = ['u_prior', 'sigma_prior', 'u_post',  'sigma_post']
        self.model_names = ['img_e', 'img_g', 'img_d']

        self.net_img_e = network.define_E(opt.image_nc, opt.ngf, opt.z_nc, opt.image_feature, opt.infer_layers,
                                          opt.down_layers, 'none', opt.G_activation, opt.init_type, opt.gpu_ids)

        self.net_img_g = network.define_G(opt.image_nc, opt.ngf, opt.z_nc, opt.image_feature, opt.r_layers, opt.down_layers,
                                          opt.output_scale, opt.norm_g, opt.G_activation, opt.init_type, opt.gpu_ids)
        # test the GAN score
        self.net_img_d = network.define_D(opt.image_nc, opt.ndf, opt.image_feature, opt.image_D_layers,
                                          opt.norm_d, opt.D_activation, opt.init_type, opt.gpu_ids)

        self.load_networks(opt.which_epoch)

    def set_input(self, input, epoch=None):
        self.input = input
        self.img = input['img']
        self.mask = input['mask']

        if len(self.gpu_ids) > 0:
            self.img = self.img.cuda(self.gpu_ids[0], async = True)
            self.mask = self.mask.cuda(self.gpu_ids[0], async = True)

    def test(self):
        self.img_truth = Variable(self.img)
        mean_value = task.img_mean_value(self.img_truth)
        img_mask = torch.where(self.mask > 0, self.img_truth, mean_value)
        self.img_truth = (self.img_truth - 0.5) * 2.0
        self.img_mask = (img_mask - 0.5) * 2.0

        self.img_real = task.scale_pyramid(self.img_truth, 4)
        mask = task.scale_pyramid(self.mask, 4)

        with torch.no_grad():
            distributions, postFeature = self.net_img_e(self.img_mask)
            mu = distributions[-1][0]
            sigma = distributions[-1][1]
            q_distribution = torch.distributions.Normal(mu, sigma)
            for i in range(self.opt.iters):
                z = q_distribution.sample()
                self.img_fake, self.atten = self.net_img_g(z, f_m=postFeature[-1], f_e=postFeature[2], mask=mask[0].chunk(3, dim=1)[0])
                self.new_img_mask = torch.where(mask[3] > 0, self.img_real[3], self.img_fake[-1])
                self.score = self.net_img_d(self.new_img_mask)
                self.save_results()

    def save_results(self):
        img_paths = self.input['img_path']

        for i in range(self.img.size(0)):
            print('process image ...... %s' % img_paths[i])
            short_path = ntpath.basename(img_paths[i])
            name = os.path.splitext(short_path)[0]
            score = torch.mean(self.score[i])
            image_name = '%s_%s.png' % (name, str(round(score.item(), 4)))
            img_path = os.path.join(self.opt.results_dir, image_name)
            # atten_name = '%s_%s_atten.png' % (name, str(round(score.item(), 4)))
            # atten_path = os.path.join(self.opt.results_dir, atten_name)
            # img_point = self.show_atten(i)
            # atten_numpy = util.tensor2im(img_point)
            # util.save_image(atten_numpy, atten_path)
            image_numpy = util.tensor2im(self.new_img_mask[i].data)
            util.save_image(image_numpy, img_path)
            mask_name = '%s_mask.png' % (name)
            mask_path = os.path.join(self.opt.results_dir, mask_name)
            mask_numpy = util.tensor2im(self.img_mask[i].data)
            util.save_image(mask_numpy, mask_path)

    def show_atten(self, index):
        x = int(32 / 2)
        y = int(32 / 2)
        atten_map, atten_weight = util.show_attention(self.atten[index].detach(), [x, y])
        # img_point = task.add_point(self.new_img_mask[index].data, [x * 8, y * 8], 4, 'blue')
        atten_map = atten_map.repeat(3,1,1)#.reshape(-1,3,32,32)
        # atten_map = F.upsample(atten_map,scale_factor=8,mode='bilinear')
        img_point = torch.where(atten_map > atten_map.mean(), atten_map*self.img_real[0][index],self.img_real[0][index]/10)
        img_point = task.add_point(img_point, [x, y ],1, 'blue')

        return img_point