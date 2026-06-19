from clip.clip import load, tokenize, available_models
from clip.idsr import idsr_diversity_loss, idsr_similarity_loss
import torch
from dataset import *
from torchvision import transforms
import argparse
import numpy as np
import random
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
from util.utils import eval_all_class
import copy

def setup_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def focal_loss(inputs, targets, alpha=-1, gamma=4, reduction="mean"):
    inputs = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def l1_loss(inputs, targets, reduction="mean"):
    return F.l1_loss(inputs, targets, reduction=reduction)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # formatter = logging.Formatter(
    #     "[%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    # )
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def print_args(logger, args):
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('{}: {}'.format(k, vars(args)[k]))
    logger.info('--------args----------\n')

def patch_alignment_loss(img_tokens, labels, gts):
    gts = gts.reshape(img_tokens[0].size(0), -1)
    labels = labels.reshape(labels.size(0), 1)
    # labels = torch.cat([labels, gts], dim=1)
    new_gts = copy.copy(gts)
    if(len(new_gts[new_gts == 0])) == 0:
        return 0
    new_gts[new_gts == 0] = -1
    b, l = new_gts.size()
    mask = torch.matmul(new_gts.reshape(b, l, 1), new_gts.reshape(b, 1, l))
    total_sim = 0
    for img_token in img_tokens:
        img_token = img_token[:, 1:, :]
        img_token = torch.nn.functional.normalize(img_token, dim=-1)
        sim = torch.matmul(img_token, img_token.permute(0, 2, 1))
        sim = sim[mask == -1].mean() - sim[mask == 1].mean()
        sim = sim if sim > 0 else 0
        total_sim = total_sim + sim
    return total_sim / len(img_tokens)

def make_exp_dir(args):
    # 格式: dataset_pl{长度}_L1_{lambda1}_L2_{lambda2}
    exp_name = "{}_pl{}_L1_{}_L2_{}_seed{}-exp{}".format(
        args.dataset, args.prompt_len, args.lambda1, args.lambda2, args.seed, args.exp
    )
    # ./train_log/mvtec_pl12_L1_1.0_L2_1.0_seed122
    exp_dir = os.path.join(args.log_dir, exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    return exp_dir

def train(args):

    # ---------------------------------------------------------
    # 1. 设置 Logger
    # ---------------------------------------------------------
    # logger = get_logger(os.path.join(args.log_dir, '[dataset]{}_[few-shot]{}_[seed]{}.txt'.format(args.dataset, args.fewshot, args.seed)))
    exp_dir = make_exp_dir(args)
    log_file_path = os.path.join(exp_dir, 'train.log')
    logger = get_logger(log_file_path)
    logger.info(f"Experiment Directory created at: {exp_dir}")
    print_args(logger, args)

    # ---------------------------------------------------------
    # 2. 模型加载与准备
    # ---------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_transform = load(name=args.model, jit = (not args.model in available_models()), device=device, download_root=args.clip_download_dir)

    clip_transform.transforms[0] = transforms.Resize(size=(args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC)
    clip_transform.transforms[1] = transforms.CenterCrop(size=(args.img_size, args.img_size))
    target_transform = transforms.Compose([
        transforms.Resize(size=clip_transform.transforms[0].size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    clip_model.eval()
    
    for param in clip_model.parameters():
        param.requires_grad_(False)
    
    clip_model = clip_model.to(device)
    clip_model.insert(args=args, tokenizer=tokenize, device=device)

    if args.dataset == 'mvtec':
        test_dataset_mvtec = MVTecDataset(root=args.data_dir, train=False, category=None, transform=clip_transform,
                                          gt_target_transform=target_transform, aug_data=True)
        test_dataset_visa = VisaDataset(root=args.data_dir, train=False, category=None, transform=clip_transform,
                                        gt_target_transform=target_transform, aug_data=False)
        train_dataset = test_dataset_mvtec
    else:
        test_dataset_mvtec = MVTecDataset(root=args.data_dir, train=False, category=None, transform=clip_transform,
                                          gt_target_transform=target_transform, aug_data=False)
        test_dataset_visa = VisaDataset(root=args.data_dir, train=False, category=None, transform=clip_transform,
                                        gt_target_transform=target_transform, aug_data=True)
        train_dataset = test_dataset_visa

    test_dataset_btad = BTADDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_dtd = DTDDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_dagm = DAGMDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
    test_data_mpdd = MPDDDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
    test_data_sdd = SDDDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
    
    all_test_dataset_dict = {
        "mvtec": test_dataset_mvtec,
        "visa": test_dataset_visa,
        "btad": test_dataset_btad,
        "dtd": test_dataset_dtd,
        'dagm': test_dataset_dagm,
        'mpdd': test_data_mpdd,
        'sdd': test_data_sdd,
    }
    if len(args.test_dataset) < 1:
        test_dataset_dict = all_test_dataset_dict
    else:
        test_dataset_dict = {}
        for ds_name in args.test_dataset:
            test_dataset_dict[ds_name] = all_test_dataset_dict[ds_name]
    if args.dataset in test_dataset_dict:
        del test_dataset_dict[args.dataset]
    # -----------------------------------------------------
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    weights_dir = os.path.join(exp_dir, 'weights')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    # ---------------------------------------------------------
    # 4. 训练逻辑 (权重加载也指向 exp_dir 或者 args.weight)
    # ---------------------------------------------------------
    if args.weight is not None:
        logger.info(f"Loading pre-trained weights from: {args.weight}")
        clip_model.idsr.load_state_dict(torch.load(os.path.join(args.weight, "{}_idsr.pt".format(args.dataset))))
        clip_model.adaptor = torch.load(os.path.join(args.weight, "{}_adaptor.pt".format(args.dataset)))
    else:
        optimizer = torch.optim.Adam(clip_model.get_trainable_parameters(), lr=args.lr, betas=(0.5, 0.999))
       
        for epoch in range(1, args.epochs + 1):
            total_loss = []
            
            for items in tqdm(train_dataloader):
                imgs, labels, gts, categories = items[:4]
                labels = labels.to(device)
                imgs = imgs.to(device)
                gts = gts.to(device)
                predict_labels, predict_masks, img_tokens = clip_model.detect_forward_seg(
                    imgs, args=args, categories=categories
                )
                gts = F.interpolate(gts, size=predict_masks[0].shape[-2:], mode='bilinear')
                gts[gts < 0.5] = 0
                gts[gts > 0.5] = 1

                # Main loss: focal + L1 + patch alignment
                loss = focal_loss(predict_labels, labels) \
                     + args.lambda1 * (focal_loss(predict_masks, gts) + l1_loss(predict_masks, gts)) \
                     + args.lambda2 * patch_alignment_loss(img_tokens, labels, gts)

                # IDSR loss: diversity + similarity (Eq.25)
                if clip_model._idsr_outputs is not None:
                    T_A_s, F_image, T_A_candidates = clip_model._idsr_outputs
                    loss_idsr = idsr_diversity_loss(T_A_s) + idsr_similarity_loss(T_A_s, F_image, T_A_candidates)
                    loss = (1.0 - args.idsr_lambda) * loss + args.idsr_lambda * loss_idsr

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())
            logger.info("Epoch: {}/{}, Loss: {:.6f}".format(epoch, args.epochs, np.mean(total_loss)))

            if epoch % args.val_freq == 0:
                # ---------------------------------------------------------
                # 4. 保存权重到 exp_dir
                # ---------------------------------------------------------

                idsr_save_path = os.path.join(weights_dir, "{}_idsr_epoch_{}.pt".format(args.dataset, epoch))
                adaptor_save_path = os.path.join(weights_dir, "{}_adaptor_epoch_{}.pt".format(args.dataset, epoch))

                idsr_latest_path = os.path.join(weights_dir, "{}_idsr_latest.pt".format(args.dataset))
                adaptor_latest_path = os.path.join(weights_dir, "{}_adaptor_latest.pt".format(args.dataset))

                torch.save(clip_model.idsr.state_dict(), idsr_save_path)
                torch.save(clip_model.adaptor, adaptor_save_path)
                torch.save(clip_model.idsr.state_dict(), idsr_latest_path)
                torch.save(clip_model.adaptor, adaptor_latest_path)

                logger.info(f"Saved weights for epoch {epoch} to {weights_dir}")
                # ---------------------------------------------------------
                # 5. 测试评估
                # ---------------------------------------------------------
                logger.info(f"Starting evaluation for epoch {epoch}...")
                for dataset_name, test_ds in test_dataset_dict.items():
                    logger.info("---------------------------{} (Epoch {})------------------------------".format(dataset_name, epoch))
                    eval_all_class(clip_model, dataset_name, test_ds, args, logger, device)
                    logger.info("-------------------------------------------------------------------------")

      
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Pytorch implemention of AF-CLIP')
    
    parser.add_argument('--clip_download_dir', type=str, default='C:/Users/Jing/.cache/clip', help='training dataset')
    
    parser.add_argument('--data_dir', type=str, default='D:/python_project/AdaCLIP/data', help='training dataset')
    
    parser.add_argument('--dataset', type=str, default='mvtec', help='training dataset', choices=['mvtec', 'visa'])
    
    parser.add_argument('--model', type=str, default="ViT-L/14@336px", help='model')
    
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    
    parser.add_argument('--lr', type=float, default=0.0001, help='learning tate')
    
    parser.add_argument('--alpha', type=float, default=0.1, help='label combination')

    parser.add_argument('--epochs', type=int, default=2, help='training epoch')
    
    parser.add_argument('--prompt_len', type=int, default=12, help='prompt length')
    
    parser.add_argument('--category', type=str, default=None, help='normal class')
    
    parser.add_argument('--fewshot', type=int, default=0, help='few shot num')
    
    parser.add_argument('--seed', type=int, default=122, help='seed')

    parser.add_argument('--exp', type=int, default=0, help='experiment num')
    
    parser.add_argument('--log_dir', type=str, default='./train_log/', help='log dir')
    
    parser.add_argument('--suffix', type=str, default='defect', help='prompt suffix')
    
    parser.add_argument('--img_size', type=int, default=518)
    
    parser.add_argument('--feature_layers', nargs='+', type=int, default=[6, 12, 18, 24], help='choose vit layers to extract features')
    
    parser.add_argument('--test_dataset', nargs='+', type=str, default=[], help='choose vit layers to extract features')
    
    parser.add_argument('--weight', type=str, default=None, help='load weight path')
    
    parser.add_argument('--vis', type=int, default=0, help='visualization results')
    
    parser.add_argument('--vis_dir', type=str, default='./vis_results/', help='visualization results dir')
    
    parser.add_argument('--memory_layers',  nargs='+', type=int, default=[6, 12, 18, 24], help='choose resnet layers to store and compare features')

    parser.add_argument('--cspf_start', type=int, default=3, help='CSPF shallow layer start (inclusive), Eq.12')

    parser.add_argument('--cspf_end', type=int, default=13, help='CSPF shallow layer end (exclusive), Eq.12')

    parser.add_argument('--lambda1', type=float, default=1, help='lambda1 for loss')
    
    parser.add_argument('--lambda2', type=float, default=1, help='lambda2 for loss')

    parser.add_argument('--val_freq', type=int, default=1, help='Frequency of validation and saving weights (in epochs)')

    parser.add_argument('--idsr_num_queries', type=int, default=4, help='IDSR number of query projections K')

    parser.add_argument('--idsr_lambda', type=float, default=0.5, help='IDSR loss weight (Eq.25: lambda for L_dis + L_sim)')

    args = parser.parse_args()
    
    args.seed = setup_seed(args.seed)
    train(args)
    
    
    
