import os
import argparse
from importlib.util import spec_from_file_location, module_from_spec
import torch
from isegm.utils.exp import init_experiment


def main():
    args = parse_args()
    
    model_script = load_module(args.model_path)
    model_base_name = getattr(model_script, 'MODEL_NAME', None)

    args.distributed = 'WORLD_SIZE' in os.environ
    cfg = init_experiment(args, model_base_name)

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    model_script.main(cfg)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', type=str, default='')
    parser.add_argument('--exp-name', type=str, default='test')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=30)
    parser.add_argument('--gpus', type=str, default='1')
    parser.add_argument('--resume-exp', type=str, default=None)
    parser.add_argument('--resume-prefix', type=str, default='latest')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--temp-model-path', type=str, default='')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--temp", type=bool, default=False)

    return parser.parse_args()


def load_module(script_path):
    spec = spec_from_file_location("model_script", script_path)
    model_script = module_from_spec(spec)
    spec.loader.exec_module(model_script)

    return model_script


if __name__ == '__main__':
    main()
