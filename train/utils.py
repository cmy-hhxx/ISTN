import os
import random
import shutil
from pathlib import Path


def move_random_samples(input_dir, testing_dir, file_extension='.tif', num_files=17):
    """
    从指定目录随机移动指定数量的文件到另一个目录。

    参数:
    input_dir (str | Path): 源文件目录。
    testing_dir (str | Path): 目标文件目录。
    file_extension (str): 要移动的文件的扩展名，默认为 '.tif'。
    num_files (int): 需要移动的文件数量，默认为 17。
    """
    input_dir = Path(input_dir)
    testing_dir = Path(testing_dir)

    # 创建目标目录如果它不存在
    testing_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有指定扩展名的文件
    files = [f for f in input_dir.iterdir() if f.suffix == file_extension]

    # 确保有足够的文件可供选择
    if len(files) < num_files:
        raise ValueError(f"目录中不足{num_files}个{file_extension}文件，请添加更多文件后再试。")

    # 随机选择指定数量的文件
    selected_files = random.sample(files, num_files)

    # 移动选定的文件
    for file in selected_files:
        shutil.move(str(file), str(testing_dir / file.name))

    print(f"已成功移动{num_files}个文件到{testing_dir}目录。")


def remove_ipynb_checkpoints(directory):
    """
    递归删除指定目录下的所有.ipynb_checkpoints文件夹，并提供反馈。

    参数:
    directory (str): 需要清理的根目录。
    """
    found_and_removed = False  # 设置一个标志，用来检测是否找到并删除了任何文件夹

    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            if dir_name == '.ipynb_checkpoints':
                full_path = os.path.join(root, dir_name)
                shutil.rmtree(full_path)
                print(f"已删除: {full_path}")
                found_and_removed = True

    if not found_and_removed:
        print("未找到任何 .ipynb_checkpoints 文件")


# move_random_samples(input_dir='/root/autodl-tmp/data/2-structures/Training/Input',testing_dir='/root/autodl-tmp/data/2-structures/Testing/Input',num_files=17)
remove_ipynb_checkpoints(directory='/root/autodl-tmp/data/2-structures')
