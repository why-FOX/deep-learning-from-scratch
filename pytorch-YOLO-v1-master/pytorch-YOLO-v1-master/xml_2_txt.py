import xml.etree.ElementTree as ET
import os
import random

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def parse_rec(filename):
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        difficult = int(obj.find('difficult').text)
        if difficult == 1:
            continue
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)
    return objects

# 路径
Annotations = r'D:\github\deep-learning-from-scratch\VOC2012\Annotations\\'
xml_files = [f for f in os.listdir(Annotations) if f.endswith('.xml')]
print(f"Total annotation files: {len(xml_files)}")

# 打乱后划分（你也可以根据 VOC 的官方分法）
random.seed(42)
random.shuffle(xml_files)
num_total = len(xml_files)
num_train = int(num_total * 0.8)
num_val = int(num_total * 0.1)
train_files = xml_files[:num_train]
val_files = xml_files[num_train:num_train + num_val]
test_files = xml_files[num_train + num_val:]

# 输出路径
output_splits = {
    'train': 'voc2012_train.txt',
    'val': 'voc2012_val.txt',
    'test': 'voc2012_test.txt'
}
splits = {
    'train': train_files,
    'val': val_files,
    'test': test_files
}

# 写文件
for split_name, file_list in splits.items():
    output_txt = output_splits[split_name]
    with open(output_txt, 'w') as txt_file:
        count_written = 0
        for xml_file in file_list:
            image_id = xml_file.split('.')[0]
            xml_path = os.path.join(Annotations, xml_file)
            results = parse_rec(xml_path)
            if len(results) == 0:
                continue
            txt_file.write(image_id + '.jpg')
            for obj in results:
                class_name = obj['name']
                class_idx = VOC_CLASSES.index(class_name)
                bbox = obj['bbox']
                txt_file.write(f' {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {class_idx}')
            txt_file.write('\n')
            count_written += 1
        print(f"[{split_name.upper()}] {count_written} annotations written to {output_txt}")
