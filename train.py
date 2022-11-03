import torch
import torch.optim as optim
import glob
import json
import argparse
import os
import random
import math
from time import gmtime, strftime
from models import *
from dataset_GBU import FeatDataLayer, DATA_LOADER
from utils import *
import torch.backends.cudnn as cudnn
import classifier_attdec

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='SUN',help='dataset: CUB, AwA2, APY, FLO, SUN')
parser.add_argument('--dataroot', default='SDGZSL_data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--image_embedding', default='res101', type=str)
parser.add_argument('--class_embedding', default='att', type=str)

parser.add_argument('--gen_nepoch', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train generater')

parser.add_argument('--zsl', type=bool, default=False, help='Evaluate ZSL or GZSL')
parser.add_argument('--finetune', type=bool, default=False, help='Use fine-tuned feature')
parser.add_argument('--ga', type=float, default=15, help='relationNet weight')
parser.add_argument('--beta', type=float, default=1, help='tc weight')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
parser.add_argument('--dis', type=float, default=3, help='Discriminator weight')
parser.add_argument('--dis_step', type=float, default=2, help='Discriminator update interval')
parser.add_argument('--kl_warmup', type=float, default=0.01, help='kl warm-up for VAE')
parser.add_argument('--tc_warmup', type=float, default=0.001, help='tc warm-up')

parser.add_argument('--vae_dec_drop', type=float, default=0.5, help='dropout rate in the VAE decoder')
parser.add_argument('--vae_enc_drop', type=float, default=0.4, help='dropout rate in the VAE encoder')
parser.add_argument('--ae_drop', type=float, default=0.2, help='dropout rate in the auto-encoder')

parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--classifier_steps', type=int, default=50, help='training steps of the classifier')

parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--nSample', type=int, default=1200, help='number features to generate per class')

parser.add_argument('--disp_interval', type=int, default=200)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--evl_interval',  type=int, default=400)
parser.add_argument('--evl_start',  type=int, default=0)
parser.add_argument('--manualSeed', type=int, default=5606, help='manual seed')

parser.add_argument('--latent_dim', type=int, default=20, help='dimention of latent z')
parser.add_argument('--q_z_nn_output_dim', type=int, default=128, help='dimention of hidden layer in encoder')
parser.add_argument('--S_dim', type=int, default=1024)
parser.add_argument('--NS_dim', type=int, default=1024)

parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--recon_weight', default=0.1, type=float, help='semantic recon loss weight')
opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ': ')))
opt.gpu = torch.device("cuda:"+opt.gpu if torch.cuda.is_available() else "cpu")


def train():
    dataset = DATA_LOADER(opt)
    opt.C_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class
    # out_dir = 'out_finetune/{}/b-{}_g-{}_lr-{}_sd-{}_dis-{}_nS-{}_nZ-{}_bs-{}_rw-{}_ft-{}'.format(opt.dataset,
    #                 opt.beta, opt.ga, opt.lr,opt.S_dim, opt.dis, opt.nSample, opt.Z_dim, opt.batchsize, opt.recon_weight, opt.finetune)
    out_dir = 'out/{}/wd-{}_b-{}_g-{}_lr-{}_sd-{}_dis-{}_nS-{}_nZ-{}_bs-{}_rw-{}'.format(opt.dataset, opt.weight_decay,
                                                                                         opt.beta, opt.ga, opt.lr,
                                                                                         opt.S_dim, opt.dis,
                                                                                         opt.nSample, opt.Z_dim,
                                                                                         opt.batchsize,
                                                                                         opt.recon_weight)
    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))

    log_dir = out_dir + '/log_{}.txt'.format(opt.dataset)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    dataset.feature_dim = dataset.train_feature.shape[1]
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class

    data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.cpu().numpy(), opt)

    opt.niter = int(dataset.ntrain/opt.batchsize) * opt.gen_nepoch

    result_gzsl_soft = Result()
    result_zsl_soft = Result()

    model = VAE(opt).to(opt.gpu)
    relationNet = RelationNet(opt).to(opt.gpu)
    discriminator = Discriminator(opt).to(opt.gpu)
    ae = AE(opt).to(opt.gpu)
    att_dec = AttributeDec(opt).to(opt.gpu)
    # print(model)

    with open(log_dir, 'a') as f:
        f.write('\n')
        f.write('Generative Model Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    relation_optimizer = optim.Adam(relationNet.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    ae_optimizer = optim.Adam(ae.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    att_dec_optimizer = optim.Adam(att_dec.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    ones = torch.ones(opt.batchsize, dtype=torch.long, device=opt.gpu)
    zeros = torch.zeros(opt.batchsize, dtype=torch.long, device=opt.gpu)
    mse = nn.MSELoss().to(opt.gpu)


    iters = math.ceil(dataset.ntrain/opt.batchsize)
    beta = 0.01
    coin = 0
    gamma = 0
    for it in range(start_step, opt.niter+1):

        if it % iters == 0:
            beta = min(opt.kl_warmup*(it/iters), 1)
            gamma = min(opt.tc_warmup * (it / iters), 1)

        blobs = data_layer.forward()
        feat_data = blobs['data']
        labels_numpy = blobs['labels'].astype(int)
        labels = torch.from_numpy(labels_numpy.astype('int')).to(opt.gpu)

        C = np.array([dataset.train_att[i,:] for i in labels])
        C = torch.from_numpy(C.astype('float32')).to(opt.gpu)
        X = torch.from_numpy(feat_data).to(opt.gpu)
        sample_C = torch.from_numpy(np.array([dataset.train_att[i, :] for i in labels.unique()])).to(opt.gpu)
        sample_C_n = labels.unique().shape[0]
        sample_label = labels.unique().cpu()

        x_mean, z_mu, z_var, z = model(X, C)
        loss, ce, kl = multinomial_loss_function(x_mean, X, z_mu, z_var, z, beta=beta)

        sample_labels = np.array(sample_label)
        re_batch_labels = []
        for label in labels_numpy:
            index = np.argwhere(sample_labels == label)
            re_batch_labels.append(index[0][0])
        re_batch_labels = torch.LongTensor(re_batch_labels)
        one_hot_labels = torch.zeros(opt.batchsize, sample_C_n).scatter_(1, re_batch_labels.view(-1, 1), 1).to(opt.gpu)

        recon_C_fake = att_dec(x_mean)
        R_cost = opt.recon_weight * WeightedL1(recon_C_fake, C)

        x1, h1, hs1, hn1 = ae(x_mean, c=recon_C_fake)
        # x1, h1, hs1, hn1 = ae(x_mean)
        relations = relationNet(hs1, sample_C)
        relations = relations.view(-1, labels.unique().cpu().shape[0])
        p_loss = opt.ga * mse(relations, one_hot_labels)

        recon_C_real = att_dec(X)
        R_cost += opt.recon_weight * WeightedL1(recon_C_real, C)

        x2, h2, hs2, hn2 = ae(X, c=recon_C_real)
        # x2, h2, hs2, hn2 = ae(X)
        relations = relationNet(hs2, sample_C)
        relations = relations.view(-1, labels.unique().cpu().shape[0])
        p_loss = p_loss + opt.ga * mse(relations, one_hot_labels)

        rec = mse(x1, X) + mse(x2, X)

        # ring_loss = ring_loss_minimizer(X, x_mean)
        if coin > 0:
            s_score = discriminator(h1)
            tc_loss = opt.beta * gamma *((s_score[:, :1] - s_score[:, 1:]).mean())
            s_score = discriminator(h2)
            tc_loss = tc_loss + opt.beta * gamma* ((s_score[:, :1] - s_score[:, 1:]).mean())

            loss = loss + p_loss + rec + tc_loss + R_cost
            coin -= 1
        else:
            s, n = permute_dims(hs1, hn1)
            b = torch.cat((s, n), 1).detach()
            s_score = discriminator(h1)
            n_score = discriminator(b)
            tc_loss = opt.dis * (F.cross_entropy(s_score, zeros) + F.cross_entropy(n_score, ones))

            s, n = permute_dims(hs2, hn2)
            b = torch.cat((s, n), 1).detach()
            s_score = discriminator(h2)
            n_score = discriminator(b)
            tc_loss = tc_loss + opt.dis * (F.cross_entropy(s_score, zeros) + F.cross_entropy(n_score, ones))

            dis_optimizer.zero_grad()
            tc_loss.backward(retain_graph=True)
            dis_optimizer.step()

            loss = loss + p_loss + rec + R_cost
            coin += opt.dis_step
        optimizer.zero_grad()
        relation_optimizer.zero_grad()
        ae_optimizer.zero_grad()
        att_dec_optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        relation_optimizer.step()
        att_dec_optimizer.step()
        ae_optimizer.step()

        if it % opt.disp_interval == 0 and it:
            log_text = 'Iter-[{}/{}]; loss: {:.3f}; kl:{:.3f}; p_loss:{:.3f}; rec:{:.3f}; R_cost:{:.3f}; tc:{:.3f}; gamma:{:.3f};'.format(it,
                                             opt.niter, loss.item(),kl.item(),p_loss.item(),rec.item(), R_cost.item(), tc_loss.item(), gamma)
            log_print(log_text, log_dir)

        if it % opt.evl_interval == 0 and it > opt.evl_start:
            model.eval()
            ae.eval()
            att_dec.eval()
            gen_feat, gen_label = synthesize_feature_test(model, ae, dataset, opt)
            gen_feat_ori, gen_label_ori = synthesize_feature_test_ori(model, dataset, opt)
            with torch.no_grad():
                train_feature = ae.encoder(dataset.train_feature.to(opt.gpu))[:,:opt.S_dim].cpu()
                test_unseen_feature = ae.encoder(dataset.test_unseen_feature.to(opt.gpu))[:,:opt.S_dim].cpu()
                test_seen_feature = ae.encoder(dataset.test_seen_feature.to(opt.gpu))[:,:opt.S_dim].cpu()

            train_X = torch.cat((train_feature, gen_feat), 0)
            train_X_ori = torch.cat((dataset.train_feature.clone(), gen_feat_ori), 0)
            test_unseen_feature_ori = dataset.test_unseen_feature.clone()
            test_seen_feature_ori = dataset.test_seen_feature.clone()

            data_ori = [train_X_ori, test_unseen_feature_ori, test_seen_feature_ori]
            train_Y = torch.cat((dataset.train_label, gen_label_ori + dataset.ntrain_class), 0)
            
            """ GZSL"""
            # att_dec=null:只使用hs训练分类器
            cls = classifier_attdec.CLASSIFIER(att_dec, data_ori, opt, train_X, train_Y, dataset, test_seen_feature, test_unseen_feature,
                                dataset.ntrain_class + dataset.ntest_class, True, opt.classifier_lr, 0.5,
                                        opt.classifier_steps, opt.nSample, True)

            result_gzsl_soft.update_gzsl(it, cls.acc_unseen, cls.acc_seen, cls.H)

            log_print("GZSL Softmax:", log_dir)
            log_print("U->T {:.2f}%  S->T {:.2f}%  H {:.2f}%  Best_H [{:.2f}% {:.2f}% {:.2f}% | Iter-{}]".format(
                cls.acc_unseen, cls.acc_seen, cls.H,  result_gzsl_soft.best_acc_U_T, result_gzsl_soft.best_acc_S_T,
                result_gzsl_soft.best_acc, result_gzsl_soft.best_iter), log_dir)

            if result_gzsl_soft.save_model:
                files2remove = glob.glob(out_dir + '/Best_model_GZSL_*')
                for _i in files2remove:
                    os.remove(_i)
                save_model(it, model,ae,att_dec, opt.manualSeed, log_text,
                           out_dir + '/Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}.tar'.format(result_gzsl_soft.best_acc,
                                                                                             result_gzsl_soft.best_acc_S_T,
                                                                                             result_gzsl_soft.best_acc_U_T))
        ###############################################################################################################

            model.train()
            ae.train()
            att_dec.train()
        if it % opt.save_interval == 0 and it:
            save_model(it, model,ae,att_dec, opt.manualSeed, log_text,
                       out_dir + '/Iter_{:d}.tar'.format(it))
            print('Save model to ' + out_dir + '/Iter_{:d}.tar'.format(it))


if __name__ == "__main__":
    train()
