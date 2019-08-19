import argparse

def get_options():
    parser = argparse.ArgumentParser(description='PyTorch face landmark Training')

    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                        help='train batchsize')

    parser.add_argument('--test-batch', default=64, type=int, metavar='N',
                        help='test batchsize')

    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')

    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='output/', type=str,
                        help='path to save checkpoint (default: checkpoint)')

    parser.add_argument('--snapshot', default='', type=str, help='pretrain model')

    args = parser.parse_args()
    return args
