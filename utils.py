import torch.nn.init as init
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def permute_dims(hs,hn):
    assert hs.dim() == 2
    assert hn.dim() == 2

    B, _ = hs.size()

    perm = torch.randperm(B).to(hs.device)
    perm_hs= hs[perm]
    perm = torch.randperm(B).to(hs.device)
    perm_hn= hn[perm]

    return perm_hs, perm_hn


def multinomial_loss_function(x_logit, x, z_mu, z_var, z, beta=1.):

    """
    Computes the cross entropy loss function while summing over batch dimension, not averaged!
    :param x_logit: shape: (batch_size, num_classes * num_channels, pixel_width, pixel_height), real valued logits
    :param x: shape (batchsize, num_channels, pixel_width, pixel_height), pixel values rescaled between [0, 1].
    :param z_mu: mean of z_0
    :param z_var: variance of z_0
    :param z_0: first stochastic latent variable
    :param z_k: last stochastic latent variable
    :param ldj: log det jacobian
    :param args: global parameter settings
    :param beta: beta for kl loss
    :return: loss, ce, kl
    """

    batch_size = x.size(0)
    target = x
    ce = nn.MSELoss(reduction='sum')(x_logit, target)
    kl = - (0.5 * torch.sum(1 + z_var.log() - z_mu.pow(2) - z_var.log().exp()))
    loss = ce + beta * kl
    loss = loss / float(batch_size)
    ce = ce / float(batch_size)
    kl = kl / float(batch_size)

    return loss, ce, kl


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S_T = 0.0
        self.best_acc_U_T = 0.0
        self.acc_list = []
        self.iter_list = []
        self.save_model = False

    def update(self, it, acc):
        self.acc_list += [acc]
        self.iter_list += [it]
        self.save_model = False
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_iter = it
            self.save_model = True

    def update_gzsl(self, it, acc_u, acc_s, H):
        self.acc_list += [H]
        self.iter_list += [it]
        self.save_model = False
        if H > self.best_acc:
            self.best_acc = H
            self.best_iter = it
            self.best_acc_U_T = acc_u
            self.best_acc_S_T = acc_s
            self.save_model = True


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.normal_(m.weight.data,  mean=0, std=0.02)
        init.constant_(m.bias, 0.0)


def log_print(s, log):
    print(s)
    with open(log, 'a') as f:
        f.write(s + '\n')


def generate_syn_feature(generator,classes, attribute,num, opt):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.X_dim)
    syn_label = torch.LongTensor(nclass*num)
    syn_att = torch.FloatTensor(num, opt.C_dim)
    syn_noise = torch.FloatTensor(num, opt.Z_dim)

    syn_att = syn_att.cuda()
    syn_noise = syn_noise.cuda()
    with torch.no_grad():

        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            with torch.no_grad():
                syn_noisev = Variable(syn_noise)
                syn_attv = Variable(syn_att)
            fake = generator.decode(syn_noisev, syn_attv)
            output = fake
            syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

def synthesize_feature_test_ori(netG, dataset, opt):
    gen_feat = torch.FloatTensor(dataset.ntest_class * opt.nSample, opt.X_dim)
    gen_label = np.zeros([0])
    with torch.no_grad():
        for i in range(dataset.ntest_class):
            text_feat = np.tile(dataset.test_att[i].astype('float32'), (opt.nSample, 1))
            text_feat = torch.from_numpy(text_feat).to(opt.gpu)
            z = torch.randn(opt.nSample, opt.Z_dim).to(opt.gpu)
            G_sample = netG.decode(z, text_feat)
            gen_feat[i*opt.nSample:(i+1)*opt.nSample] = G_sample
            gen_label = np.hstack((gen_label, np.ones([opt.nSample])*i))
    return gen_feat, torch.from_numpy(gen_label.astype(int))


def synthesize_feature_test(netG, ae, dataset, opt):
    gen_feat = torch.FloatTensor(dataset.ntest_class * opt.nSample, opt.S_dim)
    gen_label = np.zeros([0])
    with torch.no_grad():
        for i in range(dataset.ntest_class):
            text_feat = np.tile(dataset.test_att[i].astype('float32'), (opt.nSample, 1))
            text_feat = torch.from_numpy(text_feat).to(opt.gpu)
            z = torch.randn(opt.nSample, opt.Z_dim).to(opt.gpu)
            G_sample = ae.encoder(netG.decode(z, text_feat))[:,:opt.S_dim]
            gen_feat[i*opt.nSample:(i+1)*opt.nSample] = G_sample
            gen_label = np.hstack((gen_label, np.ones([opt.nSample])*i))
    return gen_feat, torch.from_numpy(gen_label.astype(int))


def save_model(it, netG, netD, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_G': netG.state_dict(),
        'state_dict_D': netD.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)

def save_model(it, model,model_d,model_dec, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'vae_state_dict': model.state_dict(),
        'disentangle_state_dict':model_d.state_dict(),
        'dec_state_dict':model_dec.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)

def WeightedL1(pred, gt):
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)
