rsync -avz --exclude 'rope/episode_*/camera_*/' rope baoyu@130.126.138.251:/mnt/nvme1n1p1/baoyu/data_rebuttal

rsync -avz rope baoyu@130.126.138.251:/mnt/nvme1n1p1/baoyu/data_rebuttal

rsync -avz --exclude 'mixed/episode_*/camera_*/' mixed zhangkaifeng@130.126.139.62:/home/zhangkaifeng/baoyu/data

rsync -av --exclude='*.png' --exclude='*.jpg' zhangkaifeng@130.126.139.62:/media/zhangkaifeng/12b6a6e6-4604-4c70-90fa-12c1d83cfd04/baoyu/data_simple/carrots_mixed_1 ./
