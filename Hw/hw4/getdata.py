import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

def download_url(url: str, save_path: Path):
    """带进度条的下载函数"""
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=save_path.name) as t:
        def reporthook(blocknum, blocksize, totalsize):
            if totalsize > 0:
                t.total = totalsize
            t.update(blocknum * blocksize - t.n)

        urllib.request.urlretrieve(url, filename=save_path, reporthook=reporthook)

def setup_ptb(data_dir: Path):
    """下载并准备 Penn Treebank 数据集"""
    ptb_url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb."
    ptb_dir = data_dir / "ptb"
    ptb_dir.mkdir(parents=True, exist_ok=True)
    
    print("--- Checking PTB Dataset ---")
    for f in ['train.txt', 'test.txt', 'valid.txt']:
        file_path = ptb_dir / f
        if not file_path.exists():
            print(f"Downloading {f}...")
            download_url(ptb_url + f, file_path)
        else:
            print(f"{f} already exists, skipping.")

def setup_cifar10(data_dir: Path):
    """下载并解压 CIFAR-10 数据集"""
    cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = data_dir / "cifar-10-python.tar.gz"
    extract_test_path = data_dir / "cifar-10-batches-py" # 解压后的标志性目录

    print("\n--- Checking CIFAR-10 Dataset ---")
    # 如果解压后的文件夹不存在，则进行下载或解压
    if not extract_test_path.exists():
        if not tar_path.exists():
            print("Downloading CIFAR-10 (approx. 170MB)...")
            download_url(cifar_url, tar_path)
        
        print(f"Extracting {tar_path.name}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        
        # 既然解压完了，由于 10-714 的工作空间通常空间有限，可以考虑删除压缩包
        # os.remove(tar_path) 
    else:
        print("CIFAR-10 already extracted, skipping.")

if __name__ == "__main__":
    # 定义基础数据目录
    BASE_DATA_DIR = Path("./data")
    BASE_DATA_DIR.mkdir(exist_ok=True)

    try:
        # setup_ptb(BASE_DATA_DIR)
        setup_cifar10(BASE_DATA_DIR)
        print("\n[Success] All datasets are ready for Needle!")
    except Exception as e:
        print(f"\n[Error] Something went wrong: {e}")