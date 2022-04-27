#%%
import os
from tqdm import tqdm
from lxml import etree
import json
import shutil
# 原始xml路径和image路径
xml_root_path = r'.\VOCdevkit\VOC2007\Annotations'
# 保存的图片和yolo格式label路径。要新建文件夹
save_label_path = r'.\VOCdevkit\VOC2007\labels'
if not os.path.exists(save_label_path):
    os.mkdir(save_label_path)
save_img_path = r'C:\Users\hubin\Desktop\mask\data\images'
def parse_xml_to_dict(xml):
    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def translate_info(file_names, class_dict):
    n = 0
    for root,dirs,files in os.walk(file_names):
        for file in tqdm(files):
            # 检查xml文件是否存在
            if 'xml' not in file:
                continue
            xml_path = os.path.join(root, file)
            # read xml
            # print(xml_path)
            with open(xml_path, encoding='utf-8') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = parse_xml_to_dict(xml)["annotation"]
            img_height = int(data["size"]["height"])
            img_width = int(data["size"]["width"])
            # img_path = data["filename"]

            # write object info into txt
            assert "object" in data.keys(), "file: '{}' lack of object key.".format(xml_path)
            if len(data["object"]) == 0:
                # 如果xml文件中没有目标就直接忽略该样本
                print("Warning: in '{}' xml, there are no objects.".format(xml_path))
                continue

            with open(os.path.join(save_label_path, file.split('.')[0] + ".txt"), "w") as f:
                for index, obj in enumerate(data["object"]):
                    # 获取每个object的box信息
                    xmin = float(obj["bndbox"]["xmin"])
                    xmax = float(obj["bndbox"]["xmax"])
                    ymin = float(obj["bndbox"]["ymin"])
                    ymax = float(obj["bndbox"]["ymax"])
                    class_name = obj["name"]
                    try:
                        class_index = class_dict.index(class_name)
                    except:
                        continue
                    # class_index = 0
                    # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
                    if xmax <= xmin or ymax <= ymin:
                        print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                        continue
                    # 将box信息转换到yolo格式
                    xcenter = xmin + (xmax - xmin) / 2
                    ycenter = ymin + (ymax - ymin) / 2
                    w = xmax - xmin
                    h = ymax - ymin
                    # 绝对坐标转相对坐标，保存6位小数
                    xcenter = round(xcenter / img_width, 6)
                    ycenter = round(ycenter / img_height, 6)
                    w = round(w / img_width, 6)
                    h = round(h / img_height, 6)
                    info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]
                    if index == 0:
                        f.write(" ".join(info))
                    else:
                        f.write("\n" + " ".join(info))

class_dict = ["LookingAround","cheat"]
translate_info(xml_root_path, class_dict)
print(len(class_dict))
#%%
import os,shutil
import numpy as np
import cv2
from tqdm import tqdm
#上一步保存的所有image和label文件路径
image_root = r'.\VOCdevkit\VOC2007\JPEGImages'
label_root = r'.\VOCdevkit\VOC2007\labels'
names = []
for root,dir,files in os.walk(label_root):
    for file in files:
        if 'txt' not in file:
            continue
        names.append(file)
val_split = 0.2
np.random.seed(10101)
np.random.shuffle(names)
num_val = int(len(names)*val_split)
num_train = len(names) - num_val
trains = names[:num_train]
vals = names[num_train:]
#保存路径
save_path_img = r'.\VOCdevkit\VOC2007\traindata'
if not os.path.exists(save_path_img):
    os.mkdir(save_path_img)
def get_train_val_data(img_root,txt_root,save_path_img,files,typ):
        def get_path(root_path,path1):
            path = os.path.join(root_path,path1)
            if not os.path.exists(path):
                os.mkdir(path)
            return path
        for val in tqdm(files):
            txt_path = os.path.join(txt_root,val)
            img_path = os.path.join(img_root,val.split('.')[0]+'.jpg')
            img_path1 = get_path(save_path_img,'images')
            txt_path1 = get_path(save_path_img,'labels')
            rt_img = get_path(img_path1,typ)
            rt_txt = get_path(txt_path1,typ)
            txt_path1 = os.path.join(rt_txt,val)
            img_path1 = os.path.join(rt_img,val.split('.')[0]+'.jpg')
            shutil.copyfile(img_path, img_path1)
            shutil.copyfile(txt_path,txt_path1)
get_train_val_data(image_root,label_root,save_path_img,vals,'val')
get_train_val_data(image_root,label_root,save_path_img,trains,'train')

#%%
# delete all file in "xml_root_path1" 
# import os
# from tqdm import tqdm

# def de(path):
#     for root,dirs,files in os.walk(path):
#         for file in tqdm(files):
#             if 'txt' in file:
#                 pa = os.path.join(root, file)
#                 os.remove(pa)

# xml_root_path1 = r'E:\python\match\dataset\319'

# de(xml_root_path1)

# %%
