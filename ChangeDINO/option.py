import argparse
import torch


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):

        self.parser.add_argument("--gpu_ids", type=str, default="2", help="gpu ids: e.g. 0. use -1 for CPU")
        self.parser.add_argument("--name", type=str, default="ESDI-15")
        self.parser.add_argument("--dataroot", type=str, default="/home/linweixuan/ChangeDINO/datasets")
        self.parser.add_argument("--dataset", type=str, default="ESDIs-SOD")
        self.parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="models are saved here")
        self.parser.add_argument("--phase", type=str, default="train")
        self.parser.add_argument("--load_pretrain", action="store_true")
        self.parser.add_argument("--save_test", action="store_true")

        self.parser.add_argument("--backbone", type=str, default="resnet34")
        self.parser.add_argument("--fpn_channels", type=int, default=128)
        self.parser.add_argument("--extract_ids", nargs="+", type=int, default=[5, 11, 17, 23])

        self.parser.add_argument("--seed", type=int, default=1)
        self.parser.add_argument("--batch_size", type=int, default=8)
        self.parser.add_argument("--num_epochs", type=int, default=100)
        self.parser.add_argument("--num_workers", type=int, default=4, help="#threads for loading data")
        self.parser.add_argument("--lr", type=float, default=1e-4)
        self.parser.add_argument("--weight_decay", type=float, default=5e-4)
        self.parser.add_argument("--epoch_val", type=int, default=60, help="从第几轮开始评测")

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(",")
        self.opt.gpu_ids = []
        for str_id in str_ids:
            gpu_id = int(str_id)
            if gpu_id >= 0:
                self.opt.gpu_ids.append(gpu_id)

        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        print("------------ Options -------------")
        for k, v in sorted(vars(self.opt).items()):
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")

        return self.opt