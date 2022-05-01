from yacs.config import CfgNode as CN
import argparse, sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Provide Q2A extraction pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/vit_b16_224",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="other opts",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()

def load_config(args):
    # Setup cfg.
    cfg = CN(new_allowed=True)
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    cfg.OUTPUT_DIR = args.cfg_file.replace("configs", "outputs").strip('.yaml')
    return cfg

def build_config():
    cfg = load_config(parse_args())
    return cfg