import torch.utils.data as data
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
import argparse
import os
import imageio


class DataSet(data.Dataset):
    def __init__(self, img_dir, resize):
        super(DataSet, self).__init__()
        self.img_paths = glob('{:s}/*'.format(img_dir))
        self.transform = transforms.Compose([transforms.Resize(size=(resize, resize))])

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item]).convert('RGB').convert('L')
        img = self.transform(img)

        return img, self.img_paths[item]

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='./data')
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--save_dir', type=str, default='./resizedata')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    dataset = DataSet(args.img_dir, args.resize)
    print('dataset:', len(dataset))

    for i in range(len(dataset)):
        img, path = dataset[i]
        path = os.path.basename(path)
        print('Processing:', path)

        imageio.imwrite(args.save_dir+'/{:s}'.format(path), img)