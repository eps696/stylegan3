import os
from tqdm import tqdm
import urllib.request
import zipfile

def get_model(url, root="./models", unzip=False):
    os.makedirs(root, exist_ok=True)
    filename = url.split('=')[-1]
    download_target = os.path.join(root, filename)
    print(download_target)

    if os.path.exists(download_target): 
        if os.path.isfile(download_target):
            raise RuntimeError(f"{download_target} exists and is not a regular file")
    else:
        with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=64, unit='iB', unit_scale=True) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break
                    output.write(buffer)
                    loop.update(len(buffer))
        if unzip:
            with zipfile.ZipFile(download_target, 'r') as zf:
                zf.extractall(root)
            os.remove(download_target)

def main():
    print(' downloading AFHQ 512x512 model with rotation')
    get_model("https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/stylegan3/1/files?redirect=true&path=stylegan3-r-afhqv2-512x512.pkl")
    print(' downloading FFHQ 1024x1024 model with rotation')
    get_model("https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/stylegan3/1/files?redirect=true&path=stylegan3-r-ffhq-1024x1024.pkl")
    print(' downloading MetFaces 1024x1024 model with rotation')
    get_model("https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/stylegan3/1/files?redirect=true&path=stylegan3-r-metfaces-1024x1024.pkl")

    # print(' downloading AFHQ 512x512 model with translation')
    # get_model("https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/stylegan3/1/files?redirect=true&path=stylegan3-t-afhqv2-512x512.pkl")
    # print(' downloading FFHQ 1024x1024 model with translation')
    # get_model("https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/stylegan3/1/files?redirect=true&path=stylegan3-t-ffhq-1024x1024.pkl")
    # print(' downloading MetFaces 1024x1024 model with translation')
    # get_model("https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/stylegan3/1/files?redirect=true&path=stylegan3-t-metfaces-1024x1024.pkl")


if __name__ == '__main__':
    main()
