import fiftyone as fo
import fiftyone.zoo as foz
import os

# 准备目录
base_dir = "./watch"
os.makedirs(base_dir, exist_ok=True)

splits = ["train"]
num_samples = 1000  # 每个分片中要下载的图片数量

for split in splits:
    dataset_name = f"open-images-v7-{split}-watch"

    # 删除已存在的同名数据集
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    # 加载数据集
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        classes=["Watch"],
        only_matching=True,
        max_samples=num_samples,
        dataset_name=dataset_name,
    )

    # 获取数据集的类型
    # print("数据集类型:", type(dataset))

    print(dataset)

    # The directory to which to write the exported dataset
    export_dir = "C:\\Users\\bearb\\Desktop\\tmp\\watch"

    # The name of the sample field containing the label that you wish to export
    # Used when exporting labeled datasets (e.g., classification or detection)
    label_field = "ground_truth"  # for example

    # The type of dataset to export
    # Any subclass of `fiftyone.types.Dataset` is supported
    dataset_type = fo.types.YOLOv5Dataset  # for example

    # Export the dataset
    dataset.export(
        export_dir=export_dir,
        dataset_type=dataset_type,
        label_field=label_field,
    )

    session = fo.launch_app(dataset)
    session.wait()


