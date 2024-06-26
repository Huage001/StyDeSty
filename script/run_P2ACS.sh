python main.py \
--task PACS \
--data_root data/PACS \
--source photo \
--target art_painting,cartoon,sketch \
--ckpt_dir checkpoint/P2ACS \
--learning_rate 0.001 \
--lr_aug 0.005 \
--iters 2000 \
--inner_iters 10 \
--alpha_feat_idt 0. \
--alpha_likelihood 0.001 \
--beta_semantic 1 \
--beta_feat_idt 0. \
--beta_likelihood 0.001 \
--network resnet18 \
--optimizer SGD \
--weight_decay 0.0005 \
--aug_weight 0.6
