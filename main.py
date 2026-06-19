"""
Copyright to ReCAP Authors, ICML 2025 Poster
built upon on Tent, EATA, DeYO and SAR code.
"""
from logging import debug
import os
import subprocess

import time
import argparse
import json
import random
import numpy as np
from pycm import *

import pickle
from collections import defaultdict

import math


from dataset.selectedRotateImageFolder import prepare_test_data
from utils.utils import get_logger
from utils.cli_utils import *


import torch    
import torch.nn.functional as F

import tent, eata, sar, deyo, recap_plpd, TCA
import MixTTA
from sam import SAM


import models.Res as Resnet
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import _load_weights
from sklearn.metrics import confusion_matrix


def validate(val_loader, model, criterion, args, mode='eval', save_model=False):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    model.eval()

    ################### Add TCA #######################            
    total,correct = 0,0
    acc_arr = []
    embeddings_arr, logits_arr = torch.tensor([]).cuda(), torch.tensor([]).cuda()
    outputs_arr,labels_arr = [],[]
    ################### Add TCA #######################
                    
    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            # compute output
            ################### Add TCA #######################
            if args.Add_TCA:
                embeddings, output = model(images)
                embeddings_arr = torch.cat([embeddings_arr, embeddings.detach()])
                logits_arr = torch.cat([logits_arr, output.detach()])
                outputs_arr.append(output.detach().cpu())
                labels_arr.append(target.cpu())
                torch.cuda.empty_cache()
            else:
                output = model(images)
            ################### Add TCA #######################
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                MixTTA.print_model_stats(model, args)

            if i > 10 and args.debug:
                break

        #### Model save ####
        if save_model:
            weight = model.model.state_dict()
            path_method = args.method
            if args.plugin_mixtta:
                path_method += '+MixTTA'
            path_corupt = 'mixed' if args.method == 'mixed_shifts' else args.corruption
            path_name = args.model+'_'+path_method+'_'+path_corupt+'_'+args.exp_type+'_'+'.pth'
            weight_path = os.path.join('./model_weight', path_name)
            os.makedirs('./model_weight', exist_ok=True)
            torch.save(weight, weight_path)
            print('Model saved: {0}'.format(weight_path))
        ###################
    
    ################### Add TCA #######################
    if args.Add_TCA:
        return top1.avg, top5.avg, embeddings_arr, logits_arr, outputs_arr, labels_arr
    else:
        return top1.avg, top5.avg
    ################### Add TCA #######################

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():

    parser = argparse.ArgumentParser(description='ReCAP exps')

    # path
    parser.add_argument('--data', default='/data/tta/ImageNet_Family/ImageNet_v1', help='path to dataset')
    parser.add_argument('--data_corruption', default='/data/tta/ImageNet_Family/ImageNet_C', help='path to corruption dataset')
    parser.add_argument('--corruption_mode', default='all', help='subset of ImageNet-C corruptions to run (default: all 15)')

    parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training.')
    parser.add_argument('--gpu', default=1, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')

    # dataloader
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--test_batch_size', default=64, type=int, help='mini-batch size for testing, before default value is 4')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')

    # corruption settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')

    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int, help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000., help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples (scaled by log(num_class) below)')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')

    # Exp Settings
    parser.add_argument('--method', default='recap_plpd', type=str, help='no_adapt, tent, eata, sar, deyo, recap_plpd')
    parser.add_argument('--model', default='resnet50_gn_timm', type=str, help='resnet50_gn_timm or resnet50_bn_torch or vitbase_timm')
    parser.add_argument('--exp_type', default='label_shifts', type=str, help='normal, mix_shifts, bs1, label_shifts')

    # SAR parameters
    parser.add_argument('--sar_margin_e0', default=0.4, type=float, help='the threshold for reliable minimization in SAR, Eqn. (2)')
    parser.add_argument('--imbalance_ratio', default=500000, type=float, help='imbalance ratio for label shift exps, selected from [1, 1000, 2000, 3000, 4000, 5000, 500000], 1  denotes totally uniform and 500000 denotes (almost the same to Pure Class Order).')
    

    # DeYO parameters
    parser.add_argument('--aug_type', default='patch', type=str, help='patch, pixel, occ')
    parser.add_argument('--occlusion_size', default=112, type=int)
    parser.add_argument('--row_start', default=56, type=int)
    parser.add_argument('--column_start', default=56, type=int)
    parser.add_argument('--deyo_margin', default=0.5, type=float,
                        help='Entropy threshold for sample selection $\tau_\mathrm{Ent}$ in Eqn. (8)')
    parser.add_argument('--deyo_margin_e0', default=0.4, type=float, help='Entropy margin for sample weighting $\mathrm{Ent}_0$ in Eqn. (10)')
    parser.add_argument('--plpd_threshold', default=0.2, type=float,
                        help='PLPD threshold for sample selection $\tau_\mathrm{PLPD}$ in Eqn. (8)')
    parser.add_argument('--patch_len', default=4, type=int, help='The number of patches per row/column')
    parser.add_argument('--fishers', default=0, type=int)
    parser.add_argument('--filter_ent', default=1, type=int)
    parser.add_argument('--filter_plpd', default=1, type=int)
    parser.add_argument('--reweight_ent', default=1, type=int)
    parser.add_argument('--reweight_plpd', default=1, type=int)
    parser.add_argument('--topk', default=1000, type=int)
    
    # ReCAP parameters
    parser.add_argument('--weight_lr', default=1.0, type=float)
    parser.add_argument('--recap_margin', default=0.8, type=float, help='Regional-Entropy threshold \tau_RE for sample selection in Eqn. (9); only samples with L_RE(x) < \tau_RE contribute to adaptation.')
    parser.add_argument('--recap_margin_L0', default=0.7, type=float, help='Reference entropy L_0 in Eqn. (9) that converts Regional-Entropy to the weighting coefficient \alpha(x) = 1 / exp(L_RE(x) - L_0).')
    parser.add_argument('--weight_tau', default=1.2, type=float, help='Region scale \tau in Eqn. (4); enlarges or shrinks the Gaussian neighborhood ')
    parser.add_argument('--weight_reg', default=0.5, type=float, help='Trade-off \lambda in Eqn. (9) between Regional Entropy L_RE and Regional Instability ')
    parser.add_argument('--reweight_threshold', default=2.0, type=float, help='Upper bound for the re-weighting coefficient \alpha(x); clips extreme values to avoid gradient explosion during adaptation.')

    # MixTTA plugin parameters
    parser.add_argument('--plugin_mixtta', default=True, type=str2bool, help='Enable MixTTA plugin')
    parser.add_argument('--layer_type', default='LoRAFC', type=str, help='Layer type: FC | LoRAFC')
    parser.add_argument('--alpha', default=4, type=int, help='LoRA scaling alpha')
    parser.add_argument('--r', default=4, type=int, help='LoRA rank r')
    parser.add_argument('--eta', default=0.9, type=float, help='Spectral Projection strength (0~1)')
    parser.add_argument('--decouple_proj', default=True, type=str2bool, help='Enable Decoupling Projection on B')
    parser.add_argument('--channel_ratio', default=0, type=float, help='Channel ratio for rank (0 to use --r)')
    parser.add_argument('--init_type', default='orthogonal', type=str, help='A init: xavier | kaiming | orthogonal')
    parser.add_argument('--lr_coeff', default=1, type=float, help='Coeff. of base learning rate')
    parser.add_argument("--tent_target_blocks", type=int, nargs="+", help="List of tent target blocks")
    parser.add_argument("--tent_target_layers", type=int, nargs="+", help="List of tent target layers")
    parser.add_argument("--tent_target_norms", type=int, nargs="+", help="List of tent target layers")
    parser.add_argument("--mixtta_target_blocks", type=int, nargs="+", help="List of mixtta target blocks")
    parser.add_argument("--mixtta_target_layers", type=int, nargs="+", help="List of mixtta target layers")
    parser.add_argument("--mixtta_target_norms", type=int, nargs="+", help="List of mixtta target layers")

    # LinearTCA parameters
    parser.add_argument('--filter_K_TCA', type=int, default=20, help="The maximum number for each category when constructing the pseudo source domain ,-1 denotes no selectiion")
    parser.add_argument('--W_num_iterations', type=int, default=20, help="The value of num_iterations during the calculation of matrix W.")
    parser.add_argument('--W_lr', type=float, default=0.001, help="The value of lr during the calculation of matrix W.")
    parser.add_argument('--Add_TCA', type=str2bool, default=False, help="Application of LinearTCA+ on top of the base method")

    return parser.parse_args()
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':


    args = get_args()

    args.num_class = 1000
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(0)

    # 현재 프로세스 PID가 어느 GPU UUID를 쓰는지
    import os as _os
    pid = str(_os.getpid())
    print("PID =", pid)
    print(subprocess.check_output([
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader"
    ]).decode())

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # Disable TF32 for full float32 precision (cuDNN TF32 is enabled by default on Ampere+)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # Deterministic cuBLAS workspace
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    # set random seeds
    if args.seed is not None:
        set_seed(args.seed)

    # console-only logging (no separate output files)
    logger = get_logger(name="project", output_directory=None, log_name=None, debug=False)
    
    
    
    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    if args.corruption_mode == 'all':
        common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    elif args.corruption_mode == 'impulse_noise':
        common_corruptions = ['impulse_noise']
    elif args.corruption_mode == 'shot_noise':
        common_corruptions = ['shot_noise'] 
    elif args.corruption_mode == 'gaussian_noise':
        common_corruptions = ['gaussian_noise'] 
    elif args.corruption_mode == 'defocus_blur':
        common_corruptions = ['defocus_blur'] 
    elif args.corruption_mode == 'glass_blur':
        common_corruptions = ['glass_blur'] 
    elif args.corruption_mode == 'motion_blur':
        common_corruptions = ['motion_blur'] 
    elif args.corruption_mode == 'zoom_blur':
        common_corruptions = ['zoom_blur'] 
    elif args.corruption_mode == 'snow':
        common_corruptions = ['snow'] 
    elif args.corruption_mode == 'frost':
        common_corruptions = ['frost']
    elif args.corruption_mode == 'fog':
        common_corruptions = ['fog']
    elif args.corruption_mode == 'contrast':
        common_corruptions = ['contrast']
    elif args.corruption_mode == 'impulse_noise':
        common_corruptions = ['impulse_noise']
    elif args.corruption_mode == 'elastic':
        common_corruptions = ['elastic_transform']
    elif args.corruption_mode == 'temp':
        common_corruptions = ['elastic_transform', 'pixelate', 'jpeg_compression']
    elif args.corruption_mode == 'jpeg':
        common_corruptions = ['jpeg_compression']

    method_list = ['tent', 'no_adapt', 'eata', 'sar', 'deyo', 'recap_plpd', 'LinearTCA']

    if args.exp_type == 'mix_shifts':
        from torch.utils.data import ConcatDataset
        datasets = []

        for cpt in common_corruptions:
            args.corruption = cpt
            logger.info(args.corruption)
            val_dataset, _ = prepare_test_data(args)
            if args.method in method_list:
                val_dataset.switch_mode(True, False)
            else:
                assert False, NotImplementedError
            datasets.append(val_dataset)

        mixed_dataset = ConcatDataset(datasets)
        logger.info(f"length of mixed dataset: {len(mixed_dataset)}")

        val_loader = torch.utils.data.DataLoader(
            mixed_dataset,
            batch_size=args.test_batch_size,
            shuffle=args.if_shuffle,
            num_workers=args.workers,
            pin_memory=True)

        common_corruptions = ['mix_shifts']

    args.e_margin *= math.log(args.num_class)
    args.sar_margin_e0 *= math.log(args.num_class)
    args.deyo_margin *= math.log(args.num_class)
    args.deyo_margin_e0 *= math.log(args.num_class)

    if args.model == 'resnet50_gn_timm':
        args.recap_margin = 0.8
        args.recap_margin_L0 = 0.8
        args.reweight_threshold = 3.0
    elif args.model == 'vitbase_timm':
        args.recap_margin = 1.0
        args.recap_margin_L0 = 1.0
        args.reweight_threshold = 1.5
    else:
        assert False, NotImplementedError

    args.recap_margin *= math.log(args.num_class)
    args.recap_margin_L0 *= math.log(args.num_class)
    args.sigmas = torch.from_numpy(np.load(f'utils/cov_{args.model}.npy'))

    # DeYO PLPD threshold (mild uses 0.3, wild scenarios use 0.2)
    args.plpd_threshold = 0.3 if args.exp_type == 'normal' else 0.2

    if args.exp_type == 'bs1':
        args.reweight_threshold = 5.0
        args.test_batch_size = 1
        logger.info("modify batch size to 1, for exp of single sample adaptation")

    if args.exp_type == 'label_shifts':
        args.if_shuffle = False
        logger.info("this exp is for label shifts, no need to shuffle the dataloader, use our pre-defined sample order")

    acc1s, acc5s = [], []
    ir = args.imbalance_ratio

    # Build the iteration list of ImageNet-C corruptions (or a single mix_shifts pass)
    loop_items = common_corruptions

    for loop_item in loop_items:
        args.corruption = loop_item

        bs = args.test_batch_size
        if args.corruption != 'mix_shifts':
            args.print_freq = 50000 // 10 // bs
        else:
            args.print_freq = 15 * 50000 // 10 // bs

        if args.method in method_list:
            if args.corruption != 'mix_shifts':
                val_dataset, val_loader = prepare_test_data(args)
                val_dataset.switch_mode(True, False)
        else:
            assert False, NotImplementedError

        # construt new dataset with online imbalanced label distribution shifts
        if args.exp_type == 'label_shifts':
            logger.info(f"imbalance ratio is {ir}")
            if args.seed == 2024:
                indices_path = './dataset/total_{}_ir_{}_class_order_shuffle_yes.npy'.format(100000, int(ir))
            else:
                indices_path = './dataset/seed{}_total_{}_ir_{}_class_order_shuffle_yes.npy'.format(args.seed, 100000, int(ir))
            logger.info(f"label_shifts_indices_path is {indices_path}")
            indices = np.load(indices_path)
            val_dataset.set_specific_subset(indices.astype(int).tolist())

        # build model for adaptation
        if args.method in method_list:
            if args.model == "resnet50_gn_timm":
                checkpoint = torch.load('./models/resnet50_gn_a1h2-8fe6c4d0.pth')
                net = timm.create_model('resnet50_gn', pretrained=False)
                net.load_state_dict(checkpoint)
                print("Model created successfully!")
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
            elif args.model == "vitbase_timm":
                net = timm.create_model("vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
                print("Model created successfully!")
                args.lr = (0.001 / 64) * bs
            elif args.model == "resnet50_bn_torch":
                net = Resnet.__dict__['resnet50'](pretrained=False)
                init = torch.load("./models/resnet50-19c8e357.pth")
                net.load_state_dict(init)
                print("Model created successfully!")
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
            else:
                assert False, NotImplementedError

            net = net.cuda()
            args.lr = args.lr * args.weight_lr
        else:
            assert False, NotImplementedError

        if args.exp_type == 'bs1' and args.method == 'sar':
            args.lr = 2 * args.lr
            logger.info("double lr for sar under bs=1")

        if args.exp_type == 'bs1' and args.method == 'deyo':
            args.lr = 2 * args.lr
            logger.info("double lr for deyo under bs=1")
        
        if args.exp_type == 'bs1' and ('recap' in args.method):
            args.lr = 2 * args.lr
            logger.info("double lr for recap under bs=1")

        
        set_seed(args.seed)

        if args.method == "tent":
            net = tent.configure_model(net)

            ################### Add TCA #######################
            if args.Add_TCA:
                if args.model == 'resnet50_gn_timm':
                    classifier_module_name = 'fc'
                elif args.model == 'vitbase_timm':
                    classifier_module_name = 'head'
                net = TCA.FeatureLogitsWrapper(net, classifier_module_name)
                params, param_names = tent.collect_params(net.model)
            else:
                params, param_names = tent.collect_params(net)
            ################### Add TCA #######################

            # logger.info(param_names)
            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9) 
            tented_model = tent.Tent(args, net, optimizer)

            ################### Add TCA #######################
            if args.Add_TCA:
                top1, top5, embeddings_arr, logits_arr, outputs_arr, labels_arr = validate(val_loader, tented_model, None, args, mode='eval', save_model=False)
            else:
                top1, top5 = validate(val_loader, tented_model, None, args, mode='eval', save_model=False)
            ################### Add TCA #######################

            ################### Add TCA #######################
            if args.Add_TCA:
                labels_arr_final = torch.cat(labels_arr, dim=0).cpu()
                label_counts = torch.bincount(labels_arr_final)
                num_classes = len(label_counts)
                proportion_vector = torch.zeros(num_classes)
                proportion_vector[0] = 1.0
                for i in range(1, num_classes):
                    proportion_vector[i] = label_counts[i].item() / label_counts[0].item()

                TCA_ = TCA.TCA(tented_model.model, filter_K=args.filter_K_TCA,
                               W_num_iterations=args.W_num_iterations, W_lr=args.W_lr)
                outputs_tca = TCA_.calculate(num_classes, embeddings_arr, logits_arr,
                                             proportion_vector).numpy()
                matrix_TCA = confusion_matrix(labels_arr_final.numpy(), outputs_tca.argmax(1))
                avg_acc_TCA = 100.0 * matrix_TCA.diagonal().sum() / matrix_TCA.sum()

                acc1 = top1
                acc5 = top5
                print('Result:')
                logger.info(f"Result under {args.corruption}. The adaptation accuracy of Tent is top1: {acc1:.1f} and top5: {acc5:.1f}")
                logger.info(f"Result under {args.corruption}. The adaptation accuracy of Tent+LinearTCA is top1: {avg_acc_TCA:.1f}")
                acc1s.append(round(float(avg_acc_TCA), 1))

            else:
                acc1 = top1
                acc5 = top5
                logger.info(f"Result under {args.corruption}. The adaptation accuracy of Tent is top1: {acc1:.1f} and top5: {acc5:.1f}")
                acc1s.append(top1.item())
                acc5s.append(top5.item())
            ################### Add TCA #######################

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")

        elif args.method == "no_adapt":
            tented_model = net
            top1, top5 = validate(val_loader, tented_model, None, args, mode='eval')
            logger.info(f"Result under {args.corruption}. Original Accuracy (no adapt) is top1: {top1:.5f} and top5: {top5:.5f}")

            acc1s.append(top1.item())
            acc5s.append(top5.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")

        elif args.method == "eata":
            # compute fisher informatrix
            _saved_corruption = args.corruption
            args.corruption = 'original'
            fisher_dataset, fisher_loader = prepare_test_data(args)
            args.corruption = _saved_corruption
            fisher_dataset.set_dataset_size(args.fisher_size)
            fisher_dataset.switch_mode(True, False)

            ################### Add TCA #######################
            if args.Add_TCA:
                if args.model == 'resnet50_gn_timm':
                    classifier_module_name = 'fc'
                elif args.model == 'vitbase_timm':
                    classifier_module_name = 'head'
                net = TCA.FeatureLogitsWrapper(net, classifier_module_name)
                params, param_names = eata.collect_params(net.model)
            else:
                params, param_names = eata.collect_params(net)
            ################### Add TCA #######################

            net = eata.configure_model(net)

            # fishers = None
            ewc_optimizer = torch.optim.SGD(params, 0.001)
            fishers = {}
            train_loss_fn = nn.CrossEntropyLoss().cuda()
            for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
                if args.gpu is not None:
                    images = images.cuda()
                if torch.cuda.is_available():
                    targets = targets.cuda()
                
                if args.Add_TCA:
                    _, outputs = net(images)
                else:
                    outputs = net(images)
                _, targets = outputs.max(1)
                loss = train_loss_fn(outputs, targets)
                loss.backward()
                for name, param in net.named_parameters():
                    if param.grad is not None:
                        if iter_ > 1:
                            fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if iter_ == len(fisher_loader):
                            fisher = fisher / iter_
                        fishers.update({name: [fisher, param.data.clone().detach()]})
                ewc_optimizer.zero_grad()
            logger.info("compute fisher matrices finished")
            del ewc_optimizer


            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            adapt_model = eata.EATA(args, net, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin)

            ################### Add TCA #######################
            if args.Add_TCA:
                top1, top5, embeddings_arr, logits_arr, outputs_arr, labels_arr = validate(val_loader, adapt_model, None, args, mode='eval', save_model=False)
            else:
                top1, top5 = validate(val_loader, adapt_model, None, args, mode='eval', save_model=False)
            ################### Add TCA #######################

            ################### Add TCA #######################
            if args.Add_TCA:
                labels_arr_final = torch.cat(labels_arr, dim=0).cpu()
                label_counts = torch.bincount(labels_arr_final)
                num_classes = len(label_counts)
                proportion_vector = torch.zeros(num_classes)
                proportion_vector[0] = 1.0
                for i in range(1, num_classes):
                    proportion_vector[i] = label_counts[i].item() / label_counts[0].item()

                TCA_ = TCA.TCA(adapt_model.model, filter_K=args.filter_K_TCA,
                               W_num_iterations=args.W_num_iterations, W_lr=args.W_lr)
                outputs_tca = TCA_.calculate(num_classes, embeddings_arr, logits_arr,
                                             proportion_vector).numpy()
                matrix_TCA = confusion_matrix(labels_arr_final.numpy(), outputs_tca.argmax(1))
                avg_acc_TCA = 100.0 * matrix_TCA.diagonal().sum() / matrix_TCA.sum()

                acc1 = top1.avg
                acc5 = top5.avg
                print('Result:')
                logger.info(f"Result under {args.corruption}. The adaptation accuracy of EATA is top1: {acc1:.1f} and top5: {acc5:.1f}")
                logger.info(f"Result under {args.corruption}. The adaptation accuracy of EATA+LinearTCA is top1: {avg_acc_TCA:.1f}")
                acc1s.append(round(float(avg_acc_TCA), 1))

            else:
                acc1 = top1
                acc5 = top5
                logger.info(f"Result under {args.corruption}. The adaptation accuracy of EATA is top1: {acc1:.1f} and top5: {acc5:.1f}")
                acc1s.append(top1.item())
                acc5s.append(top5.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")

        elif args.method in ['sar']:
            ################### Add TCA #######################
            if args.Add_TCA:
                total,correct = 0,0
                acc_arr = []
                embeddings_arr, logits_arr = torch.tensor([]).cuda(), torch.tensor([]).cuda()
                outputs_arr,labels_arr = [],[]
                if args.model == 'resnet50_gn_timm':
                    classifier_module_name = 'fc'
                elif args.model == 'vitbase_timm':
                    classifier_module_name = 'head'
                net = TCA.FeatureLogitsWrapper(net, classifier_module_name)
            ################### Add TCA #######################

            net = sar.configure_model(net)

            ################### Add TCA #######################
            if args.Add_TCA:
                params, param_names = sar.collect_params(net.model)
            else:
                params, param_names = sar.collect_params(net)
            ################### Add TCA #######################

            logger.info(param_names)

            base_optimizer = torch.optim.SGD
            optimizer = SAM(params, base_optimizer, lr=args.lr, momentum=0.9)
            adapt_model = sar.SAR(args, net, optimizer, margin_e0=args.sar_margin_e0)

            batch_time = AverageMeter('Time', ':6.3f')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')
            progress = ProgressMeter(
                len(val_loader),
                [batch_time, top1, top5],
                prefix='Test: ')
            end = time.time()
            for i, dl in enumerate(val_loader):
                images, target = dl[0], dl[1]
                if args.gpu is not None:
                    images = images.cuda()
                if torch.cuda.is_available():
                    target = target.cuda()
                ################### Add TCA #######################
                if args.Add_TCA:
                    embeddings, output = adapt_model(images)
                else:
                    output = adapt_model(images)
                ################### Add TCA #######################
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                ################### Add TCA #######################
                if args.Add_TCA:
                    embeddings_arr = torch.cat([embeddings_arr, embeddings.detach()])
                    logits_arr = torch.cat([logits_arr, output.detach()])
                    outputs_arr.append(output.detach().cpu())
                    labels_arr.append(target)
                    torch.cuda.empty_cache()
                ################### Add TCA #######################

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    MixTTA.print_model_stats(adapt_model.model, args)
                    MixTTA.clear_logging_stats(adapt_model.model)
                    progress.display(i)

            ################### Add TCA #######################
            if args.Add_TCA:
                labels_arr_final = torch.cat(labels_arr, dim=0).cpu()
                label_counts = torch.bincount(labels_arr_final)
                num_classes = len(label_counts)
                proportion_vector = torch.zeros(num_classes)
                proportion_vector[0] = 1.0
                for i in range(1, num_classes):
                    proportion_vector[i] = label_counts[i].item() / label_counts[0].item()

                TCA_ = TCA.TCA(adapt_model.model, filter_K=args.filter_K_TCA,
                               W_num_iterations=args.W_num_iterations, W_lr=args.W_lr)
                outputs_tca = TCA_.calculate(num_classes, embeddings_arr, logits_arr,
                                             proportion_vector).numpy()
                matrix_TCA = confusion_matrix(labels_arr_final.numpy(), outputs_tca.argmax(1))
                avg_acc_TCA = 100.0 * matrix_TCA.diagonal().sum() / matrix_TCA.sum()

                acc1 = top1.avg
                acc5 = top5.avg
                print('Result:')
                logger.info(f"Result under {args.corruption}. The adaptation accuracy of SAR is top1: {acc1:.1f} and top5: {acc5:.1f}")
                logger.info(f"Result under {args.corruption}. The adaptation accuracy of SAR+LinearTCA is top1: {avg_acc_TCA:.1f}")
                acc1s.append(round(float(avg_acc_TCA), 1))

            else:
                acc1 = top1.avg
                acc5 = top5.avg
                logger.info(f"Result under {args.corruption}. The adaptation accuracy of SAR is top1: {acc1:.1f} and top5: {acc5:.1f}")
                acc1s.append(top1.avg.item())
                acc5s.append(top5.avg.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")

        elif args.method in ['deyo']:
            ################### Add TCA #######################
            if args.Add_TCA:
                total,correct = 0,0
                acc_arr = []
                embeddings_arr, logits_arr = torch.tensor([]).cuda(), torch.tensor([]).cuda()
                outputs_arr,labels_arr = [],[]
                if args.model == 'resnet50_gn_timm':
                    classifier_module_name = 'fc'
                elif args.model == 'vitbase_timm':
                    classifier_module_name = 'head'
                net = TCA.FeatureLogitsWrapper(net, classifier_module_name)
            ################### Add TCA #######################
            biased = (args.exp_type=='spurious')

            net = deyo.configure_model(net)
            ################### Add TCA #######################
            if args.Add_TCA:
                params, param_names = deyo.collect_params(net.model)
            else:
                params, param_names = deyo.collect_params(net)
            ################### Add TCA #######################
            logger.info(param_names)

            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            adapt_model = deyo.DeYO(net, args, optimizer, deyo_margin=args.deyo_margin, margin_e0=args.deyo_margin_e0)

            batch_time = AverageMeter('Time', ':6.3f')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')

            if biased:
                LL_AM = AverageMeter('LL Acc', ':6.2f')
                LS_AM = AverageMeter('LS Acc', ':6.2f')
                SL_AM = AverageMeter('SL Acc', ':6.2f')
                SS_AM = AverageMeter('SS Acc', ':6.2f')
                progress = ProgressMeter(
                    len(val_loader),
                    [batch_time, top1, top5, LL_AM, LS_AM, SL_AM, SS_AM],
                    prefix='Test: ')
            else:
                progress = ProgressMeter(
                    len(val_loader),
                    [batch_time, top1, top5],
                    prefix='Test: ')
            
            end = time.time()
            count_backward = 1e-6
            final_count_backward =1e-6
            count_corr_pl_1 = 0
            count_corr_pl_2 = 0
            total_count_backward = 1e-6
            total_final_count_backward =1e-6
            total_count_corr_pl_1 = 0
            total_count_corr_pl_2 = 0
            correct_count = [0,0,0,0]
            total_count = [1e-6,1e-6,1e-6,1e-6]

            for i, dl in enumerate(val_loader):
                images, target = dl[0], dl[1]
                if args.gpu is not None:
                    images = images.cuda()
                if torch.cuda.is_available():
                    target = target.cuda()
                if biased:
                    place = dl[2].cuda()
                    group = 2*target + place
                else:
                    group=None

                ################### Add TCA #######################
                if args.Add_TCA:
                    embeddings, output, backward, final_backward, corr_pl_1, corr_pl_2 = adapt_model(images, i, target, group=group)
                else:
                    output, backward, final_backward, corr_pl_1, corr_pl_2 = adapt_model(images, i, target, group=group)
                ################### Add TCA #######################
                
                if biased:
                    TFtensor = (output.argmax(dim=1)==target)
                    
                    for group_idx in range(4):
                        correct_count[group_idx] += TFtensor[group==group_idx].sum().item()
                        total_count[group_idx] += len(TFtensor[group==group_idx])
                    acc1, acc5 = accuracy(output, target, topk=(1, 1))
                else:
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                
                count_backward += backward
                final_count_backward += final_backward
                total_count_backward += backward
                total_final_count_backward += final_backward
                
                count_corr_pl_1 += corr_pl_1
                count_corr_pl_2 += corr_pl_2
                total_count_corr_pl_1 += corr_pl_1
                total_count_corr_pl_2 += corr_pl_2

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                
                if i % args.print_freq == 0:
                    if biased:
                        LL = correct_count[0]/total_count[0]*100
                        LS = correct_count[1]/total_count[1]*100
                        SL = correct_count[2]/total_count[2]*100
                        SS = correct_count[3]/total_count[3]*100
                        LL_AM.update(LL, images.size(0))
                        LS_AM.update(LS, images.size(0))
                        SL_AM.update(SL, images.size(0))
                        SS_AM.update(SS, images.size(0))
                    
                    count_backward = 1e-6
                    final_count_backward =1e-6
                    count_corr_pl_1 = 0
                    count_corr_pl_2 = 0

                ################### Add TCA #######################
                if args.Add_TCA:
                    embeddings_arr = torch.cat([embeddings_arr, embeddings.detach()])
                    logits_arr = torch.cat([logits_arr, output.detach()])
                    outputs_arr.append(output.detach().cpu())
                    labels_arr.append(target)
                    torch.cuda.empty_cache()
                ################### Add TCA #######################

                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)
                    MixTTA.print_model_stats(adapt_model.model, args)
                    MixTTA.clear_logging_stats(adapt_model.model)

            ################### Add TCA #######################
            if args.Add_TCA:
                labels_arr_final = torch.cat(labels_arr, dim=0).cpu()
                label_counts = torch.bincount(labels_arr_final)
                num_classes = len(label_counts)
                proportion_vector = torch.zeros(num_classes)
                proportion_vector[0] = 1.0
                for i in range(1, num_classes):
                    proportion_vector[i] = label_counts[i].item() / label_counts[0].item()

                TCA_ = TCA.TCA(adapt_model.model, filter_K=args.filter_K_TCA,
                               W_num_iterations=args.W_num_iterations, W_lr=args.W_lr)
                outputs_tca = TCA_.calculate(num_classes, embeddings_arr, logits_arr,
                                             proportion_vector).numpy()
                matrix_TCA = confusion_matrix(labels_arr_final.numpy(), outputs_tca.argmax(1))
                avg_acc_TCA = 100.0 * matrix_TCA.diagonal().sum() / matrix_TCA.sum()

                acc1 = top1.avg
                acc5 = top5.avg
                print('Result:')
                logger.info(f"Result under {args.corruption}. The adaptation accuracy of DeYO is top1: {acc1:.1f} and top5: {acc5:.1f}")
                logger.info(f"Result under {args.corruption}. The adaptation accuracy of DeYO+LinearTCA is top1: {avg_acc_TCA:.1f}")
                acc1s.append(round(float(avg_acc_TCA), 1))

            else:
                acc1 = top1.avg
                acc5 = top5.avg
                logger.info(f"Result under {args.corruption}. The adaptation accuracy of DeYO is top1: {acc1:.1f} and top5: {acc5:.1f}")
                acc1s.append(top1.avg.item())
                acc5s.append(top5.avg.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")

        elif args.method in ['recap_plpd']:
            net = recap_plpd.configure_model(net)
            params, param_names = recap_plpd.collect_params(net)
            logger.info(param_names)
            labels_arr = []  # collected for TCA proportion_vector

            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9) 
            adapt_model = recap_plpd.ReCAP(net, args, optimizer, \
                margin = args.recap_margin, \
                margin_L0 = args.recap_margin_L0, \
                weight_reg = args.weight_reg, reweight_threshold = args.reweight_threshold, \
                sigmas=args.sigmas, batch_size = bs, weight_tau = args.weight_tau)
            

            batch_time = AverageMeter('Time', ':6.3f')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')

            progress = ProgressMeter(
                len(val_loader),
                [batch_time, top1, top5],
                prefix='Test: ')

            end = time.time()
            start_time = time.time()

            for i, dl in enumerate(val_loader):
                images, target = dl[0], dl[1]
                if args.gpu is not None:
                    images = images.cuda()
                if torch.cuda.is_available():
                    target = target.cuda() 

                output = adapt_model(images)
                if args.Add_TCA:
                    labels_arr.append(target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # measure elapsed time

                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)
                    MixTTA.print_model_stats(adapt_model.model, args)
                    MixTTA.clear_logging_stats(adapt_model.model)

            acc1 = top1.avg
            acc5 = top5.avg

            if args.Add_TCA:
                outputs_tca = adapt_model.compute_tca_output(labels_arr)
                labels_arr_final = torch.cat(labels_arr).cpu().numpy()
                matrix_TCA = confusion_matrix(labels_arr_final, outputs_tca.numpy().argmax(1))
                avg_acc_TCA = 100.0 * matrix_TCA.diagonal().sum() / matrix_TCA.sum()
                adapt_model.reset_tca_buffer()

                logger.info(f"Result under {args.corruption}. The adaptation accuracy of ReCAP+plpd is top1: {acc1:.1f} and top5: {acc5:.1f}")
                logger.info(f"Result under {args.corruption}. The adaptation accuracy of ReCAP+plpd+LinearTCA is top1: {avg_acc_TCA:.1f}")
                acc1s.append(round(float(avg_acc_TCA), 1))
            else:
                logger.info(f"Result under {args.corruption}. The adaptation accuracy of ReCAP+plpd is top1: {acc1:.1f} and top5: {acc5:.1f}")
                acc1s.append(top1.avg.item())
                acc5s.append(top5.avg.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")

        elif args.method in ['LinearTCA']:
            if args.model == 'resnet50_gn_timm':
                classifier_module_name = 'fc'
            elif args.model == 'vitbase_timm':
                classifier_module_name = 'head'
            adapt_model = TCA.FeatureLogitsWrapper(net, classifier_module_name)
            adapt_model.cuda()    
            total,correct = 0,0
            acc_arr = []
            embeddings_arr, logits_arr = torch.tensor([]).cuda(), torch.tensor([]).cuda()
            outputs_arr,labels_arr = [],[]

            batch_time = AverageMeter('Time', ':6.3f')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')

            progress = ProgressMeter(
                len(val_loader),
                [batch_time],
                prefix='Test: ')

            end = time.time()
            start_time = time.time()

            for idx, sample in enumerate(val_loader):
                image,label = sample
                image = image.cuda()
                with torch.no_grad():
                    embeddings, logits = adapt_model(image)
                embeddings_arr = torch.cat([embeddings_arr, embeddings.detach()])
                logits_arr = torch.cat([logits_arr, logits.detach()])
                outputs_arr.append(logits.detach().cpu())
                labels_arr.append(label)
                torch.cuda.empty_cache()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % args.print_freq == 0:
                    progress.display(idx)
                
            # no-adapt baseline accuracy
            outputs_np = torch.cat(outputs_arr, 0).numpy()
            labels_np  = torch.cat(labels_arr).numpy()
            avg_acc_TTA = 100.0 * (outputs_np.argmax(1) == labels_np).mean()

            labels_arr_final = torch.cat(labels_arr, dim=0)
            label_counts = torch.bincount(labels_arr_final)
            num_classes = len(label_counts)
            proportion_vector = torch.zeros(num_classes)
            proportion_vector[0] = 1.0
            for i in range(1, num_classes):
                proportion_vector[i] = label_counts[i].item() / label_counts[0].item()

            TCA_ = TCA.TCA(adapt_model, filter_K=args.filter_K_TCA,
                           W_num_iterations=args.W_num_iterations, W_lr=args.W_lr)
            outputs_tca = TCA_.calculate(num_classes, embeddings_arr, logits_arr,
                                         proportion_vector).numpy()
            matrix_TCA  = confusion_matrix(labels_arr_final.numpy(), outputs_tca.argmax(1))
            avg_acc_TCA = 100.0 * matrix_TCA.diagonal().sum() / matrix_TCA.sum()

            print('Result:')
            print(f'  TTA Accuracy:     {avg_acc_TTA:.2f}')
            print(f'  LinearTCA Accuracy: {avg_acc_TCA:.2f}')

            logger.info(f"Result under {args.corruption}. The adaptation accuracy of LinearTCA is top1: {float(avg_acc_TCA):.5f}")
            
            acc1s.append(round(float(avg_acc_TCA), 1))
            logger.info(f"acc1s are {acc1s}")

        else:
            assert False, NotImplementedError

