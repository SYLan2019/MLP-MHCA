from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import cv2
import torch
import numpy as np

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

from pysot_toolkit.toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, VOTDataset, NFSDataset, VOTLTDataset
from pysot_toolkit.toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, EAOBenchmark, F1Benchmark
from pysot_toolkit.toolkit.visualization import draw_success_precision
from pysot_toolkit.trackers.tracker import Tracker
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone
from pysot_toolkit.bbox import get_axis_aligned_bbox
from pysot_toolkit.toolkit.datasets import DatasetFactory

import optuna
import logging


parser = argparse.ArgumentParser(description='mlp-mhca tune')
parser.add_argument('--dataset', '-d', type=str, default='OTB2015',
                    help='dataset name')
parser.add_argument('--num', '-n', default=1, type=int,
                    help='number of thread to eval')
parser.add_argument('--tracker_prefix', '-t', default='mlp-mhca',
                    type=str, help='tracker name')
parser.add_argument("--gpu_id", default="0", type=str, help="gpu id")
args = parser.parse_args()


def eval(tracker_name):

    tracker_dir = "./"
    trackers = [tracker_name]

    if 'OTB' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'LaSOT' == args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = EAOBenchmark(dataset)
        eval_eao = benchmark.eval(tracker_name)
        eao = eval_eao[tracker_name]['all']
        return eao

    return 0


# fitness function
def objective(trial):
    # different params
    Window_Influence = trial.suggest_uniform('window_influence', 0.050, 0.650)
    Window_Penalty = trial.suggest_uniform('penalty_k', 0.000, 0.600)
    Scale_LR = trial.suggest_uniform('scale_lr', 0.100, 0.800)
    
    # rebuild tracker
    net_path = './workSpace/checkpoints/ltr/mlp-mhca/mlp-mhca' #Absolute path of the model
    net = NetWithBackbone(net_path=net_path, use_gpu=True)
    tracker = Tracker(name='mlp-mhca', net=net, window_penalty=Window_Penalty, window_influence=Window_Influence, scale_lr=Scale_LR, exemplar_size=128, instance_size=256)

    model_name = net_path.split('/')[-1]
    tracker_name = os.path.join('tune_results',args.dataset, model_name + \
                    '_wi-{:.3f}'.format(Window_Influence) + \
                    '_pk-{:.3f}'.format(Window_Penalty) + \
                    '_lr-{:.3f}'.format(Scale_LR))

    if args.dataset in ['OTB100']:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - w / 2, cy - h / 2, w, h]
                    init_info = {'init_bbox': gt_bbox_}
                    tracker.initialize(img, init_info)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['target_bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                                          'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                                           '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path,
                                           '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                if not os.path.isdir(tracker_name):
                    os.makedirs(tracker_name)
                result_path = os.path.join(tracker_name, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, video.name, toc, idx / toc))
        auc = eval(tracker_name)
        info = "{:s}, window_influence: {:1.17f}, penalty_k: {:1.17f}, scale_lr: {:1.17f}, AUC: {:1.3f}".format(
            model_name, Window_Influence, Window_Penalty, Scale_LR, auc)
        logging.getLogger().info(info)
        print(info)
        return auc


if __name__ == "__main__":

    torch.set_num_threads(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # create dataset
    dataset_root = './pysot_toolkit/testing_dataset/OTB100'
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    tune_result = os.path.join('tune_results', args.dataset)
    if not os.path.isdir(tune_result):
                os.makedirs(tune_result)
    net_path = './workSpace/checkpoints/ltr/mlp-mhca/mlp-mhca'
    log_path = os.path.join(tune_result, (net_path).split('/')[-1] + '.log')
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.FileHandler(log_path))
    optuna.logging.enable_propagation()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10000)
    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))


