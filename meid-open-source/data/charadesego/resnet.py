import os
import time
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
# import enviroments
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['CUDA_VISIBLE_DEVICES']='9'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# config
# vis = True
vis = False
vis_row = 4

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

classes = ["ants", "bees"]


def img_transform(img_rgb, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_rgb: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_rgb)
    return img_t


def get_img_name(img_dir, format="jpg"):
    """
    获取文件夹下format格式的文件名
    :param img_dir: str
    :param format: str
    :return: list
    """
    file_names = os.listdir(img_dir)
    # 使用 list(filter(lambda())) 筛选出 jpg 后缀的文件
    img_names = list(filter(lambda x: x.endswith(format), file_names))

    if len(img_names) < 1:
        raise ValueError("{}下找不到{}格式数据".format(img_dir, format))
    img_names.sort()
    return img_names


def get_model(m_path, vis_model=False):

    resnet101 = models.resnet101(pretrained=True)
    resnet101 = torch.nn.Sequential(*(list(resnet101.children())[:-1]))

    # 修改全连接层的输出
    # num_ftrs = resnet101.fc.in_features
    # resnet101.fc = nn.Linear(num_ftrs, 2)

    # 加载模型参数
    # checkpoint = torch.load(m_path)
    # resnet101.load_state_dict(checkpoint['model_state_dict'])

    if vis_model:
        from torchsummary import summary
        summary(resnet101, input_size=(3, 224, 224), device="cpu")

    return resnet101


if __name__ == "__main__":
    # tr = open('train_lt.txt','r').readlines()
    # te = open('test_lt.txt','r').readlines()
    va = open('val_lt.txt','r').readlines()
    root = '/data/lxj/data/CharadesEgo/CharadesEgo_v1_rgb/'
    import numpy as np
    import os
    import pdb
    import math
    # save_dir = '/data/lxj/data/charades/charades_lt_r101/'
    save_dir = '/data/lxj/data/CharadesEgo/r101/'
    exist_file=os.listdir(save_dir)
    # 2. model
    model_path = "./checkpoint_14_epoch.pkl"
    resnet101 = get_model(model_path, False)
    resnet101.to(device)
    resnet101.eval()
    from tqdm import tqdm
    # for kk in [tr,te]:
    for kk in [va]:
        for i in tqdm(kk):
            i = i.strip().split(' ')[0]
            if i+'.npy' in exist_file:
                continue
            # pdb.set_trace()
            path = root+i
            # img_dir = os.path.join(enviroments.hymenoptera_data_dir,"val/bees")
            img_dir = path
            time_total = 0
            img_list, img_pred = list(), list()

            # 1. data
            img_names = get_img_name(img_dir)
            num_img = len(img_names)
            # flag=math.ceil(num_img/200)
            flag=math.floor(num_img/200)
            count=0
            final=[]
            with torch.no_grad():
                for idx, img_name in enumerate(img_names):
                    
                    if count%flag!=0:
                        count+=1
                        continue
                    count+=1
                    path_img = os.path.join(img_dir, img_name)

                    # step 1/4 : path --> img
                    img_rgb = Image.open(path_img).convert('RGB')

                    # step 2/4 : img --> tensor
                    img_tensor = img_transform(img_rgb, inference_transform)
                    img_tensor.unsqueeze_(0)
                    img_tensor = img_tensor.to(device)

                    # step 3/4 : tensor --> vector
                    time_tic = time.time()
                    outputs = resnet101(img_tensor)
                    time_toc = time.time()

                    # step 4/4 : visualization
                    # _, pred_int = torch.max(outputs.data, 1)
                    # pred_str = classes[int(pred_int)]
                    outputs=outputs.reshape(2048)
                    outputs=outputs.cpu().numpy()
                    final.append(outputs)
                    # pdb.set_trace()

                    if vis:
                        img_list.append(img_rgb)
                        img_pred.append(pred_str)

                        if (idx+1) % (vis_row*vis_row) == 0 or num_img == idx+1:
                            for i in range(len(img_list)):
                                plt.subplot(vis_row, vis_row, i+1).imshow(img_list[i])
                                plt.title("predict:{}".format(img_pred[i]))
                            plt.show()
                            plt.close()
                            img_list, img_pred = list(), list()

                    time_s = time_toc-time_tic
                    time_total += time_s

                    print('{:d}/{:d}: {} {:.3f}s '.format(idx + 1, num_img, img_name, time_s))

            print("\ndevice:{} total time:{:.1f}s mean:{:.3f}s".
                format(device, time_total, time_total/num_img))
            if torch.cuda.is_available():
                print("GPU name:{}".format(torch.cuda.get_device_name()))
            
            # pdb.set_trace()
            np.save(save_dir+i+'.npy', final)
            