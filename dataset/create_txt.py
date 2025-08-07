import os, random, glob
random.seed(0)
import numpy as np

percent = 0.01

base_path = os.getcwd()
print(base_path)

image_listdir = [f'{base_path}/images/train/{i}' for i in os.listdir(f'{base_path}/images/train')]
print(f'labeled 1 percent size:{int(len(image_listdir) * percent)}')
with open('train_1_percent.txt', 'w+') as f:
    f.write('\n'.join(image_listdir[:int(len(image_listdir) * percent)]))

image_listdir = image_listdir[int(len(image_listdir) * percent):]
print(f'unlabeled 1 percent size:{len(image_listdir)}')
with open('unlabeled_1_percent.txt', 'w+') as f:
    f.write('\n'.join(image_listdir))

image_listdir = [f'{base_path}/images/val/{i}' for i in os.listdir(f'{base_path}/images/val')]
with open('val.txt', 'w+') as f:
    f.write('\n'.join(image_listdir))

image_listdir = [f'{base_path}/images/test/{i}' for i in os.listdir(f'{base_path}/images/test')]
with open('test.txt', 'w+') as f:
    f.write('\n'.join(image_listdir))
