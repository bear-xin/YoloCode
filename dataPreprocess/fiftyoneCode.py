import fiftyone as fo
import fiftyone.zoo as foz
import os
import logging



#------------------------------------------------------------------------------getDataset
#--------------------------------------------dataset1 from train
split = "train"
classes = ["Watch"]
num_samples = 1000  # 每个分片中要下载的图片数量
dataset_name = f"open-images-v7-{split}-watch"
# 删除已存在的同名数据集
if fo.dataset_exists(dataset_name):
    fo.delete_dataset(dataset_name)
# 加载数据集
dataset1 = foz.load_zoo_dataset(
    "open-images-v7",
    split=split,
    label_types=["detections"],
    classes=classes,
    only_matching=True,
    max_samples=num_samples,
    dataset_name=dataset_name,
)
print(dataset1)

#-----------------------------dataset2 from test
split = "test"
num_samples = 1000  # 每个分片中要下载的图片数量
dataset_name = f"open-images-v7-{split}-watch"
# 删除已存在的同名数据集
if fo.dataset_exists(dataset_name):
    fo.delete_dataset(dataset_name)
# 加载数据集
dataset2 = foz.load_zoo_dataset(
    "open-images-v7",
    split=split,
    label_types=["detections"],
    classes=classes,
    only_matching=True,
    max_samples=num_samples,
    dataset_name=dataset_name,
)
print(dataset2)
# -----------------------------dataset3 from validation
split = "validation"
num_samples = 1000  # 每个分片中要下载的图片数量
dataset_name = f"open-images-v7-{split}-watch"
# 删除已存在的同名数据集
if fo.dataset_exists(dataset_name):
    fo.delete_dataset(dataset_name)
# 加载数据集
dataset3 = foz.load_zoo_dataset(
    "open-images-v7",
    split=split,
    label_types=["detections"],
    classes=classes,
    only_matching=True,
    max_samples=num_samples,
    dataset_name=dataset_name,
)
print(dataset3)


#------------------------------------------------------------------------------mergeDataset
# #---------------------------------------------------预处理，统一标签
# # 将dataset2中的"Watch"类别名称改为"watch"
# for sample in dataset2:
#     for detection in sample.ground_truth.detections:
#         if detection.label == "Watch":
#             detection.label = "watch"
#     sample.save()
# 创建新的合并后的数据集
merged_dataset = dataset1.clone()  # 克隆dataset1以保留其数据结构
merged_dataset.add_samples(dataset2)  # 添加来自dataset2的样本
merged_dataset.add_samples(dataset3)  # 添加来自dataset2的样本


#------------------------------------------------------------------------------exportDataset
# The directory to which to write the exported dataset
export_dir = "C:\\Users\\bearb\\Desktop\\GraduationProject\\datasets\\Watch\\"

# The name of the sample field containing the label that you wish to export
# Used when exporting labeled datasets (e.g., classification or detection)
label_field = "ground_truth"  # for example

# The type of dataset to export
# Any subclass of `fiftyone.types.Dataset` is supported
dataset_type = fo.types.YOLOv5Dataset  # for example

# Export the dataset
merged_dataset.export(
    export_dir=export_dir,
    dataset_type=dataset_type,
    label_field=label_field,
)

#--------------------------------------------------------------------打印日志文件
# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler(export_dir+"log.txt", mode="w"),
                        logging.StreamHandler()
                    ])
# 使用日志用法
# logging.debug("这是一个debug信息")
# logging.info("这是一个info信息")
# logging.warning("这是一个warning信息")
# logging.error("这是一个error信息")
# logging.critical("这是一个critical信息")

#--------------------------------------------------------------------统计数据（注意，这里统计的是每个类label的数量，不是图片张数）
# 获取类别标签统计
label_counts = merged_dataset.count_values("ground_truth.detections.label")

# 打印数据集信息
logging.info(dataset1)
logging.info(dataset2)
logging.info(dataset3)
logging.info(merged_dataset)

# 打印每个类别及其对应的样本数量
for label, count in label_counts.items():
    print(f"{label}: {count}")
    logging.info(f"{label}: {count}")

#-------------------------------------------------------------------打开可视化页面
session = fo.launch_app(merged_dataset)
session.wait()
