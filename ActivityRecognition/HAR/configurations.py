import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--im_size', type=int, default=512)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--ratio', type=float, default=2.0, help='ratio between context box and bounding box')
parser.add_argument('--scale', type=float, default=5.0, help='scale of resized original image over image')
# parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--dataset', type=str, default='construction')
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--text', type=str, default='', help='notes for experiment')
parser.add_argument('--resume', type=str, default="", help='resume training or not')
parser.add_argument('--mask', type=str2bool, default=False, help='masking the cbox by bbox or not')
parser.add_argument('--use_feature_maps', type=str2bool, default=True, help='use feature maps for classify or not')
parser.add_argument('--use_self_attn', type=str2bool, default=True, help='use self attention or not')
parser.add_argument('--modified', type=str2bool, default=False, help='use modified resnet to have larger feature maps or not')
#parser.add_argument('--max_epoch', type=int, default=50000)

parser.add_argument('--max_epoch', type=int, default=20)

parser.add_argument('--log_interval', type=int, default=1)
parser.add_argument('--test', type=str2bool, default=False)
parser.add_argument('--output', type=str, default="features/")
parser.add_argument('--val_step', type=int, default=5)
parser.add_argument('--exclude_bg', type=str2bool, default=False)
args = parser.parse_args()