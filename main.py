import argparse
import os
import random
import numpy as np
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn
import network
import time
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
    parser.add_argument("--scheduler", default='linear', type=str, help="Learning rate scheduler")
    parser.add_argument("--alpha_feat_idt", default=1., type=float)
    parser.add_argument("--alpha_likelihood", default=1., type=float)
    parser.add_argument("--beta_semantic", default=1., type=float)
    parser.add_argument("--beta_likelihood", default=0.1, type=float)
    parser.add_argument("--beta_feat_idt", default=1., type=float)
    parser.add_argument("--lr_aug", default=0.005, type=float)
    parser.add_argument("--aug_weight", default=0.6, type=float)
    parser.add_argument("--weight_decay", default=0.0005, type=float)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--test_freq", type=int, default=50)
    parser.add_argument("--print_freq", type=int, default=10)
    return parser.parse_args()


def main(args):
    device = torch.device(args.device)
    name = str(int(time.time()))
    print('Running ID: %s' % name)
    os.makedirs(os.path.join(args.ckpt_dir, name), exist_ok=True)
    save_options(os.path.join(args.ckpt_dir, name, 'options.txt'), args)

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
    else:
        raise NotImplementedError

    train_set = get_datasets(args.task, args.data_root, domains=args.source, is_train=True)
    test_sets = get_datasets(args.task, args.data_root, domains=args.target, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.n_workers,
                                               pin_memory=True,
                                               sampler=torch.utils.data.RandomSampler(
                                                   train_set, replacement=True, num_samples=2 ** 31 - 1))
    eval_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.n_workers,
                                              pin_memory=True,
                                              sampler=torch.utils.data.RandomSampler(
                                                  train_set, replacement=True, num_samples=2 ** 31 - 1))
    train_iter = iter(train_loader)
    eval_iter = iter(eval_loader)
    test_loaders = [torch.utils.data.DataLoader(test_set, batch_size=args.batch_size * 2, shuffle=False,
                                                num_workers=args.n_workers,
                                                pin_memory=True, drop_last=False) for test_set in test_sets]

    backbone, classifier = network.get_network(args.network, n_classes)
    backbone = backbone.to(device)
    classifier = classifier.to(device)
    if args.optimizer == 'SGD':
        optimizer_backbone = torch.optim.SGD(backbone.parameters(), lr=args.learning_rate,
                                             nesterov=args.nesterov, momentum=0.9, weight_decay=args.weight_decay)
        optimizer_classifier = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate,
                                               nesterov=args.nesterov, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer_backbone = torch.optim.Adam(backbone.parameters(), lr=args.learning_rate,
                                              weight_decay=args.weight_decay)
        optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate,
                                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    if args.scheduler == 'linear':
        scheduler_classifier = torch.optim.lr_scheduler.StepLR(optimizer_classifier, step_size=int(args.iters * 0.45))
        scheduler_backbone = torch.optim.lr_scheduler.StepLR(optimizer_backbone, step_size=int(args.iters * 0.45))
    elif args.scheduler == 'cos':
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_classifier, args.iters)
        scheduler_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_backbone, args.iters)
    else:
        raise NotImplementedError
    optimizer_aug = torch.optim.SGD(aug_net.parameters(), lr=args.lr_aug)

    best_acc = 0.
    best_msg = ''
    all_losses = []
    for i in range(args.iters):

        losses = {}
        data = eval_iter.__next__()
        image = data[0].to(device)
        label = data[1].to(device)
        optimizer_backbone.zero_grad()
        image_aug = normalize(torch.sigmoid(aug_net(image))) * args.aug_weight + image * (1. - args.aug_weight)
        images = torch.cat([image_aug, image], dim=0)
        labels = torch.cat([label, label], dim=0)
        feat = backbone(images)
        pred, log_var, mu, embed = classifier(feat)
        losses['feat_idt'] = F.mse_loss(feat[:label.size(0)], feat[label.size(0):]) * args.alpha_feat_idt
        losses['likelihood'] = likelihood(mu[label.size(0):], log_var[label.size(0):],
                                          embed[:label.size(0)]) * args.alpha_likelihood
        losses['outer_cls'] = F.cross_entropy(pred, labels)
        emb_aug = F.normalize(embed[:label.size(0)]).unsqueeze(1)
        emb_src = F.normalize(embed[label.size(0):]).unsqueeze(1)
        loss = losses['outer_cls'] + losses['feat_idt'] + losses['likelihood']
        loss.backward()
        optimizer_backbone.step()

        optimizer_aug.zero_grad()
        image_aug = normalize(torch.sigmoid(aug_net(image, True))) * args.aug_weight + image * (1. - args.aug_weight)
        images = torch.cat([image_aug, image], dim=0)
        feat = backbone(images)
        pred, log_var, mu, embed = classifier(feat)
        losses['adv_feat_idt'] = F.mse_loss(feat[:label.size(0)], feat[label.size(0):]) * args.beta_feat_idt
        losses['adv_likelihood'] = club(mu[label.size(0):], log_var[label.size(0):],
                                        embed[:label.size(0)]) * args.beta_likelihood
        losses['adv_cls'] = F.cross_entropy(pred, labels)
        losses['semantic'] = mmd_rbf(embed[:label.size(0)], embed[label.size(0):]) * args.beta_semantic
        loss = losses['semantic'] + losses['adv_likelihood'] - losses['adv_feat_idt'] - losses['adv_cls']
        loss.backward()
        optimizer_aug.step()

        for j in range(args.inner_iters):
            data = train_iter.__next__()
            image = data[0].to(device)
            label = data[1].to(device)
            optimizer_classifier.zero_grad()
            with torch.no_grad():
                image_aug = normalize(torch.sigmoid(aug_net(image))) * args.aug_weight + image * (1. - args.aug_weight)
                images = torch.cat([image_aug, image], dim=0)
                labels = torch.cat([label, label], dim=0)
                feat = backbone(images)
            pred, _, _, _ = classifier(feat)
            losses['inner_cls'] = F.cross_entropy(pred, labels)
            losses['inner_cls'].backward()
            optimizer_classifier.step()

        scheduler_backbone.step()
        scheduler_classifier.step()

        all_losses.append({k: v.item() for k, v in losses.items()})

        if (i + 1) % args.print_freq == 0:
            msg = 'iteration %04d' % (i + 1)
            for k, v in losses.items():
                msg += ' loss_%s: %.3f' % (k, v.item())
            print(msg)

        if (i + 1) % args.test_freq == 0:
            backbone.eval()
            classifier.eval()
            with torch.no_grad():
                acc = []
                for test_loader in test_loaders:
                    cur_total = 0
                    cur_correct = 0
                    for data in test_loader:
                        image = data[0].to(device)
                        label = data[1].to(device)
                        feat = backbone(image)
                        pred, _, _, _ = classifier(feat)
                        pred_class = torch.argmax(pred, 1)
                        cur_total += label.shape[0]
                        cur_correct += torch.eq(pred_class, label).sum().item()
                    acc.append(cur_correct / cur_total)
            backbone.train()
            classifier.train()
            mean_acc = sum(acc) * 100 / len(acc)
            msg = 'Test Accuracy: %.2f' % mean_acc
            for item in acc:
                msg += '\t[%.2f]' % (item * 100)
            print(msg)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_msg = msg
                torch.save(classifier.state_dict(), os.path.join(args.ckpt_dir, name, 'classifier_best.pth'))
                torch.save(backbone.state_dict(), os.path.join(args.ckpt_dir, name, 'backbone_best.pth'))
                torch.save(aug_net.state_dict(), os.path.join(args.ckpt_dir, name, 'aug_best.pth'))
                print('Best Model Saved!')
    print('Best %s' % best_msg)
    torch.save(all_losses, os.path.join(args.ckpt_dir, name, 'all_losses.pth'))
    torch.save(classifier.state_dict(), os.path.join(args.ckpt_dir, name, 'classifier_final.pth'))
    torch.save(backbone.state_dict(), os.path.join(args.ckpt_dir, name, 'backbone_final.pth'))
    torch.save(aug_net.state_dict(), os.path.join(args.ckpt_dir, name, 'aug_final.pth'))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    config = get_args()
    main(config)
