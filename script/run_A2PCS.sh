python main.py \
--task PACS \
--data_root data/PACS \
--source art_painting \
--target photo,cartoon,sketch \
--ckpt_dir checkpoint/A2PCS \
--learning_rate 0.001 \
--lr_aug 0.005 \
--iters 2000 \
--inner_iters 10 \
--alpha_feat_idt 0.1 \
--network alexnet \
--optimizer SGD \
--weight_decay 0.0005 \
--aug_weight 0.6
