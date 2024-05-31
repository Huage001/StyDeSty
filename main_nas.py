import argparse
import os
import random
import numpy as np
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn
import network_nas as network
from torchvision import transforms
from dataset import get_datasets
from util import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Task", default='PACS')
    parser.add_argument("--data_root", type=str, help="Data root", default='data/PACS')
    parser.add_argument("--source", type=str, help="Source", default='photo')
    parser.add_argument("--target", type=str, help="Target", default='art_painting,cartoon,sketch')
    parser.add_argument("--ckpt_dir", type=str, help="Path of saving checkpoint", default='checkpoint/P2ACS')
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--iters", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--inner_iters", type=int, default=10, help="Number of inner training iterations")
    parser.add_argument("--network", help="Which network to use", default="resnet18")
    parser.add_argument("--optimizer", help='Which optimizer to use, Adam or SGD', default='SGD')
    parser.add_argument("--nesterov", default=True, type=bool, help="Use nesterov")
    parser.add_argument("--alpha_feat_idt", default=1., type=float)
    parser.add_argument("--alpha_likelihood", default=1., type=float)
    parser.add_argument("--alpha_con", default=0., type=float)
    parser.add_argument("--beta_semantic", default=1., type=float)
    parser.add_argument("--beta_likelihood", default=0.1, type=float)
    parser.add_argument("--beta_feat_idt", default=1., type=float)
    parser.add_argument("--lr_aug", default=0.005, type=float)
    parser.add_argument("--aug_weight", default=0.6, type=float)
    parser.add_argument("--weight_decay", default=0.0005, type=float)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--test_freq", type=int, default=1000)
    parser.add_argument("--print_freq", type=int, default=10)
    return parser.parse_args()


def main(args):
    device = torch.device(args.device)
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.task == 'PACS':
        aug_net = network.AugNet().to(device)
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)
        n_classes = 7
    elif args.task == 'Digits':
        aug_net = network.AugNetSmall().to(device)
        normalize = transforms.Normalize([0, 0, 0], [1, 1, 1]).to(device)
        n_classes = 10
    elif args.task == 'CIFAR10-C':
        aug_net = network.AugNetSmall().to(device)
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).to(device)
        n_classes = 10
    elif args.task == 'OfficeHome':
        aug_net = network.AugNet().to(device)
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)
        n_classes = 65
    else:
        raise NotImplementedError

    train_set = get_datasets(args.task, args.data_root, domains=args.source, is_train=True)
    test_sets = get_datasets(args.task, args.data_root, domains=args.target, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.n_workers,
                                               pin_memory=True,
                                               sampler=torch.utils.data.RandomSampler(
                                                   train_set, replacement=True, num_samples=2 ** 31 - 1))
    train_iter = iter(train_loader)
    test_loaders = [torch.utils.data.DataLoader(test_set, batch_size=args.batch_size * 2, shuffle=False,
                                                num_workers=args.n_workers,
                                                pin_memory=True, drop_last=False) for test_set in test_sets]

    model, norm_logit = network.get_network(args.network, n_classes)
    model = model.to(device)
    norm_logit = norm_logit.to(device)
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                    nesterov=args.nesterov, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    optimizer_norm = torch.optim.SGD([norm_logit.requires_grad_()], lr=args.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.iters * 0.45))
    optimizer_aug = torch.optim.SGD(aug_net.parameters(), lr=args.lr_aug)

    best_acc = 0.
    best_msg = ''
    for i in range(args.iters):

        losses = {}
        data = train_iter.__next__()
        image = data[0].to(device)
        label = data[1].to(device)
        optimizer.zero_grad()
        optimizer_norm.zero_grad()
        in_weights = diff_argmax(norm_logit)
        image_aug = normalize(torch.sigmoid(aug_net(image))) * args.aug_weight + image * (1. - args.aug_weight)
        images = torch.cat([image_aug, image], dim=0)
        labels = torch.cat([label, label], dim=0)
        pred, in_feats, log_var, mu, embed = model(images, in_weights)
        losses['feat_idt'] = sum([F.mse_loss(in_feat[:label.size(0)],
                                             in_feat[label.size(0):]) * in_weight * args.alpha_feat_idt
                                  for in_weight, in_feat in zip(in_weights, in_feats)])
        losses['outer_cls'] = F.cross_entropy(pred, labels)
        loss = losses['outer_cls'] + losses['feat_idt']
        loss.backward()
        optimizer.step()
        optimizer_norm.step()

        optimizer_aug.zero_grad()
        in_weights = diff_argmax(norm_logit)
        image_aug = normalize(torch.sigmoid(aug_net(image, estimation=True))) * args.aug_weight + image * (1. - args.aug_weight)
        images = torch.cat([image_aug, image], dim=0)
        pred, in_feats, log_var, mu, embed = model(images, in_weights)
        losses['adv_likelihood'] = club(mu[label.size(0):], log_var[label.size(0):],
                                        embed[:label.size(0)]) * args.beta_likelihood
        losses['adv_cls'] = F.cross_entropy(pred, labels)
        losses['semantic'] = mmd_rbf(embed[:label.size(0)], embed[label.size(0):]) * args.beta_semantic
        loss = losses['semantic'] + losses['adv_likelihood'] - losses['adv_cls']
        loss.backward()
        optimizer_aug.step()

        scheduler.step()

        if (i + 1) % args.print_freq == 0:
            msg = 'iteration %04d' % (i + 1)
            for k, v in losses.items():
                msg += ' loss_%s: %.3f' % (k, v.item())
            print(msg)
            print(str(norm_logit.argmax().item()) + '\t' + str(list(norm_logit.data.cpu().numpy())))

        if (i + 1) % args.test_freq == 0:
            model.eval()
            with torch.no_grad():
                acc = []
                in_weights = diff_argmax(norm_logit)
                for test_loader in test_loaders:
                    cur_total = 0
                    cur_correct = 0
                    for data in test_loader:
                        image = data[0].to(device)
                        label = data[1].to(device)
                        pred, _, _, _, _ = model(image, in_weights)
                        pred_class = torch.argmax(pred, 1)
                        cur_total += label.shape[0]
                        cur_correct += torch.eq(pred_class, label).sum().item()
                    acc.append(cur_correct / cur_total)
            model.train()
            mean_acc = sum(acc) * 100 / len(acc)
            msg = 'Test Accuracy: %.2f' % mean_acc
            for item in acc:
                msg += '\t[%.2f]' % (item * 100)
            print(msg)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_msg = msg
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'model_best.pth'))
                print('Best Model Saved!')

    print('Best %s' % best_msg)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    config = get_args()
    main(config)
