import os, glob

base_path = "./"


for file in glob.glob(f"{base_path}/CIFAR_*"):
    # file = os.path.join(base_path, f"{i}.mp4.tar.gz")
    # rename = os.path.join(base_path, f"old_{i}.mp4.tar.gz")
    rename = file.rsplit("/", 1)[1]
    # rename = os.path.join(base_path, f"nonnormalized_{rename}")
    rename = os.path.join(base_path, f"{rename.replace('CIFAR', 'cifar10')}")
    print(file, rename)
    try:
        os.rename(file, rename)
    except FileNotFoundError:
        print("file does not exist", file, rename)
