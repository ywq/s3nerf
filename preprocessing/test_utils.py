import os
import torch
from models import model_utils
from utils import eval_utils, time_utils 
import numpy as np
from PIL import Image
import scipy

def get_itervals(args, split):
    if split not in ['train', 'val', 'test']:
        split = 'test'
    args_var = vars(args)
    disp_intv = args_var[split+'_disp']
    save_intv = args_var[split+'_save']
    stop_iters = args_var['max_'+split+'_iter']
    return disp_intv, save_intv, stop_iters

def test(args, split, loader, models, log, epoch, recorder):
    models[0].eval()
    models[1].eval()
    log.printWrite('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync)

    disp_intv, save_intv, stop_iters = get_itervals(args, split)
    res = []
    light_dirs, light_ints = [],[]
    with torch.no_grad():
        for i, sample in enumerate(loader):
            data = model_utils.parseData(args, sample, timer, split)
            input = model_utils.getInput(args, data)

            pred_c = models[0](input); timer.updateTime('Forward')
            input.append(pred_c)
            pred = models[1](input); timer.updateTime('Forward')

            recoder, iter_res, error = prepareRes(args, data, pred_c, pred, recorder, log, split)

            res.append(iter_res)
            iters = i + 1

            if iters % save_intv == 0:
                light_dirs.append(pred_c['dirs'])

            if stop_iters > 0 and iters >= stop_iters: break

    light_dirs = torch.stack(light_dirs, dim=0).detach().cpu().numpy()
    np.save(os.path.join(args.log_dir, 'light_direction_sdps.npy'), light_dirs)

def prepareRes(args, data, pred_c, pred, recorder, log, split):
    data_batch = args.val_batch if split == 'val' else args.test_batch
    iter_res = []
    error = ''
    if args.s1_est_d:
        l_acc, data['dir_err'] = eval_utils.calDirsAcc(data['dirs'].data, pred_c['dirs'].data, data_batch)
        recorder.updateIter(split, l_acc.keys(), l_acc.values())
        iter_res.append(l_acc['l_err_mean'])
        error += 'D_%.3f-' % (l_acc['l_err_mean']) 
    if args.s1_est_i:
        int_acc, data['int_err'] = eval_utils.calIntsAcc(data['ints'].data, pred_c['intens'].data, data_batch)
        recorder.updateIter(split, int_acc.keys(), int_acc.values())
        iter_res.append(int_acc['ints_ratio'])
        error += 'I_%.3f-' % (int_acc['ints_ratio'])

    if args.s2_est_n:
        acc, error_map = eval_utils.calNormalAcc(data['n'].data, pred['n'].data, data['m'].data)
        recorder.updateIter(split, acc.keys(), acc.values())
        iter_res.append(acc['n_err_mean'])
        error += 'N_%.3f-' % (acc['n_err_mean'])
        data['error_map'] = error_map['angular_map']
    return recorder, iter_res, error

def prepareSave(args, data, pred_c, pred):
    results = [data['img'].data, data['m'].data, (data['n'].data+1) / 2]
    if args.s2_est_n:
        pred_n = (pred['n'].data + 1) / 2
        masked_pred = pred_n * data['m'].data.expand_as(pred['n'].data)
        res_n = [masked_pred, data['error_map']]
        results += res_n

    nrow = data['img'].shape[0]
    return results, nrow
