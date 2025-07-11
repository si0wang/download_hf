import os
import time
import requests
from requests.adapters import HTTPAdapter, Retry
from huggingface_hub import configure_http_backend
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='name of huggingface model to download', default='russwang/ThinkLite-VL-70k')
parser.add_argument('--path', type=str, help='the parent directory of the download output', default='./')
parser.add_argument('--max_retry', type=int, help='max retries for downloading', default=10)
parser.add_argument('--proxy', type=lambda x: 'y' in x.lower(), help="whether enable proxy", default=False)
parser.add_argument('--proxy_addr', type=str, help='specify the proxy address', default='http://127.0.0.1:7890')
args = parser.parse_args()

if args.proxy:
    if 'http_proxy' in os.environ:
        args.proxy_addr = os.environ['http_proxy']

print(f"proxy = {args.proxy}, addr = {args.proxy_addr}")


# Create a factory function that returns a Session with configured proxies
def backend_factory() -> requests.Session:
    # Define a Session configured to retry 5 times with backoff of 0.5s, 1s, 2s, 4s,...
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1))
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    if args.proxy:
        session.proxies = {'http': args.proxy_addr, 'https': args.proxy_addr}

    return session


if __name__ == '__main__':
    if args.proxy:
        print(f"enable proxy, addr = {args.proxy_addr}")
        configure_http_backend(backend_factory=backend_factory)
        os.environ['http_proxy'] = args.proxy_addr
        os.environ['https_proxy'] = args.proxy_addr

    from huggingface_hub import snapshot_download

    failed = 0
    start_time = time.time()
    while failed <= args.max_retry:
        try:
            model = snapshot_download(
                repo_id=args.model,
                repo_type = "dataset",
                resume_download=True,
                force_download=True,
                local_dir=args.path,
                local_dir_use_symlinks=False,
                cache_dir=args.path
            )
        except Exception as e:
            failed += 1
            sleep_time = int(min(1.5 ** failed, 45))
            print(f"failed, fail/max_retry = {failed}/{args.max_retry}, sleeping {sleep_time}")
            time.sleep(sleep_time)
            continue

        print(f"download, done! failed = {failed}, time_cost = {time.time() - start_time:.1f} seconds")
        exit(0)

    print("failed!")
    exit(1)
