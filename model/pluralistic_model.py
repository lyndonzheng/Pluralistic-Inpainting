import torch
from .base_model import BaseModel
from . import network, base_function, external_function
from util import task
import itertools


class Pluralistic(BaseModel):
    """This class implements the pluralistic image completion, for 256*256 resolution image inpainting"""
    def name(self):
        return "Pluralistic Image Completion"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--output_scale', type=int, default=4, help='# of number of the output scale')
        if is_train:
            parser.add_argument('--train_paths', type=str, default='two', help='training strategies with one path or two paths')
            parser.add_argument('--lambda_rec', type=float, default=20.0, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_kl', type=float, default=20.0, help='weight for kl divergence loss')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for generation loss')

        return parser

    def __init__(self, opt):
        """Initial the pluralistic model"""
        BaseModel.__init__(self, opt)

        self.loss_names = ['kl_rec', 'kl_g', 'app_rec', 'app_g', 'ad_g', 'img_d', 'ad_rec', 'img_d_rec']
        self.visual_names = ['img_m', 'img_c', 'img_truth', 'img_out', 'img_g', 'img_rec']
        self.value_names = ['u_m', 'sigma_m', 'u_post', 'sigma_post', 'u_prior', 'sigma_prior']
        self.model_names = ['E', 'G', 'D', 'D_rec']
        self.distribution = []

        # define the inpainting model
        self.net_E = network.define_e(ngf=32, z_nc=128, img_f=128, layers=5, norm='none', activation='LeakyReLU',
                                      init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_G = network.define_g(ngf=32, z_nc=128, img_f=128, L=0, layers=5, output_scale=opt.output_scale,
                                      norm='instance', activation='LeakyReLU', init_type='orthogonal', gpu_ids=opt.gpu_ids)
        # define the discriminator model
        self.net_D = network.define_d(ndf=32, img_f=128, layers=5, model_type='ResDis', init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_D_rec = network.define_d(ndf=32, img_f=128, layers=5, model_type='ResDis', init_type='orthogonal', gpu_ids=opt.gpu_ids)

        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.GANLoss(opt.gan_mode)
            self.L1loss = torch.nn.L1Loss()
            self.L2loss = torch.nn.MSELoss()
            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_G.parameters()),
                        filter(lambda p: p.requires_grad, self.net_E.parameters())), lr=opt.lr, betas=(0.0, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_D.parameters()),
                                                filter(lambda p: p.requires_grad, self.net_D_rec.parameters())),
                                                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        # load the pretrained model and schedulers
        self.setup(opt)

    def set_input(self, input, epoch=0):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input
        self.image_paths = self.input['img_path']
        self.img = input['img']
        self.mask = input['mask']

        if len(self.gpu_ids) > 0:
            self.img = self.img.cuda(self.gpu_ids[0], async=True)
            self.mask = self.mask.cuda(self.gpu_ids[0], async=True)

        # get I_m and I_c for image with mask and complement regions for training
        self.img_truth = self.img * 2 - 1
        self.img_m = self.mask * self.img_truth
        self.img_c = (1 - self.mask) * self.img_truth

        # get multiple scales image ground truth and mask for training
        self.scale_img = task.scale_pyramid(self.img_truth, self.opt.output_scale)
        self.scale_mask = task.scale_pyramid(self.mask, self.opt.output_scale)

    def test(self):
        """Forward function used in test time"""
        # save the groundtruth and masked image
        self.save_results(self.img_truth, data_name='truth')
        self.save_results(self.img_m, data_name='mask')

        # encoder process
        distribution, f = self.net_E(self.img_m)
        q_distribution = torch.distributions.Normal(distribution[-1][0], distribution[-1][1])
        scale_mask = task.scale_img(self.mask, size=[f[2].size(2), f[2].size(3)])

        # decoder process
        for i in range(self.opt.nsampling):
            z = q_distribution.sample()
            self.img_g, attn = self.net_G(z, f_m=f[-1], f_e=f[2], mask=scale_mask.chunk(3, dim=1)[0])
            self.img_out = (1 - self.mask) * self.img_g[-1].detach() + self.mask * self.img_m
            self.score = self.net_D(self.img_out)
            self.save_results(self.img_out, i, data_name='out')

    def get_distribution(self, distributions):
        """Calculate encoder distribution for img_m, img_c"""
        # get distribution
        sum_valid = (torch.mean(self.mask.view(self.mask.size(0), -1), dim=1) - 1e-5).view(-1, 1, 1, 1)
        m_sigma = 1 / (1 + ((sum_valid - 0.8) * 8).exp_())
        p_distribution, q_distribution, kl_rec, kl_g = 0, 0, 0, 0
        self.distribution = []
        for distribution in distributions:
            p_mu, p_sigma, q_mu, q_sigma = distribution
            # the assumption distribution for different mask regions
            m_distribution = torch.distributions.Normal(torch.zeros_like(p_mu), m_sigma * torch.ones_like(p_sigma))
            # m_distribution = torch.distributions.Normal(torch.zeros_like(p_mu), torch.ones_like(p_sigma))
            # the post distribution from mask regions
            p_distribution = torch.distributions.Normal(p_mu, p_sigma)
            p_distribution_fix = torch.distributions.Normal(p_mu.detach(), p_sigma.detach())
            # the prior distribution from valid region
            q_distribution = torch.distributions.Normal(q_mu, q_sigma)

            # kl divergence
            kl_rec += torch.distributions.kl_divergence(m_distribution, p_distribution)
            if self.opt.train_paths == "one":
                kl_g += torch.distributions.kl_divergence(m_distribution, q_distribution)
            elif self.opt.train_paths == "two":
                kl_g += torch.distributions.kl_divergence(p_distribution_fix, q_distribution)
            self.distribution.append([torch.zeros_like(p_mu), m_sigma * torch.ones_like(p_sigma), p_mu, p_sigma, q_mu, q_sigma])

        return p_distribution, q_distribution, kl_rec, kl_g

    def get_G_inputs(self, p_distribution, q_distribution, f):
        """Process the encoder feature and distributions for generation network"""
        f_m = torch.cat([f[-1].chunk(2)[0], f[-1].chunk(2)[0]], dim=0)
        f_e = torch.cat([f[2].chunk(2)[0], f[2].chunk(2)[0]], dim=0)
        scale_mask = task.scale_img(self.mask, size=[f_e.size(2), f_e.size(3)])
        mask = torch.cat([scale_mask.chunk(3, dim=1)[0], scale_mask.chunk(3, dim=1)[0]], dim=0)
        z_p = p_distribution.rsample()
        z_q = q_distribution.rsample()
        z = torch.cat([z_p, z_q], dim=0)
        return z, f_m, f_e, mask

    def forward(self):
        """Run forward processing to get the inputs"""
        # encoder process
        distributions, f = self.net_E(self.img_m, self.img_c)
        p_distribution, q_distribution, self.kl_rec, self.kl_g = self.get_distribution(distributions)

        # decoder process
        z, f_m, f_e, mask = self.get_G_inputs(p_distribution, q_distribution, f)
        results, attn = self.net_G(z, f_m, f_e, mask)
        self.img_rec = []
        self.img_g = []
        for result in results:
            img_rec, img_g = result.chunk(2)
            self.img_rec.append(img_rec)
            self.img_g.append(img_g)
        self.img_out = (1-self.mask) * self.img_g[-1].detach() + self.mask * self.img_truth

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss +=gradient_penalty

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D, self.net_D_rec)
        self.loss_img_d = self.backward_D_basic(self.net_D, self.img_truth, self.img_g[-1])
        self.loss_img_d_rec = self.backward_D_basic(self.net_D_rec, self.img_truth, self.img_rec[-1])

    def backward_G(self):
        """Calculate training loss for the generator"""

        # encoder kl loss
        self.loss_kl_rec = self.kl_rec.mean() * self.opt.lambda_kl * self.opt.output_scale
        self.loss_kl_g = self.kl_g.mean() * self.opt.lambda_kl * self.opt.output_scale

        # generator adversarial loss
        base_function._freeze(self.net_D, self.net_D_rec)
        # g loss fake
        D_fake = self.net_D(self.img_g[-1])
        self.loss_ad_g = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        # rec loss fake
        D_fake = self.net_D_rec(self.img_rec[-1])
        D_real = self.net_D_rec(self.img_truth)
        self.loss_ad_rec = self.L2loss(D_fake, D_real) * self.opt.lambda_g

        # calculate l1 loss ofr multi-scale outputs
        loss_app_rec, loss_app_g = 0, 0
        for i, (img_rec_i, img_fake_i, img_real_i, mask_i) in enumerate(zip(self.img_rec, self.img_g, self.scale_img, self.scale_mask)):
            loss_app_rec += self.L1loss(img_rec_i, img_real_i)
            if self.opt.train_paths == "one":
                loss_app_g += self.L1loss(img_fake_i, img_real_i)
            elif self.opt.train_paths == "two":
                loss_app_g += self.L1loss(img_fake_i*mask_i, img_real_i*mask_i)
        self.loss_app_rec = loss_app_rec * self.opt.lambda_rec
        self.loss_app_g = loss_app_g * self.opt.lambda_rec

        # if one path during the training, just calculate the loss for generation path
        if self.opt.train_paths == "one":
            self.loss_app_rec = self.loss_app_rec * 0
            self.loss_ad_rec = self.loss_ad_rec * 0
            self.loss_kl_rec = self.loss_kl_rec * 0

        total_loss = 0

        for name in self.loss_names:
            if name != 'img_d' and name != 'img_d_rec':
                total_loss += getattr(self, "loss_" + name)

        total_loss.backward()

    def optimize_parameters(self):
        """update network weights"""
        # compute the image completion results
        self.forward()
        # optimize the discrinimator network parameters
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # optimize the completion network parameters
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
