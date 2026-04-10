
rlaunch --gpu=2 --memory=409600 --cpu=32 \
	--charged-group=steai_gpu \
	--private-machine=group \
	--mount=gpfs://gpfs1/gaoao-p:/mnt/shared-storage-user/gaoao-p \
	--mount=gpfs://gpfs1/tanxin:/mnt/shared-storage-user/tanxin \
	--mount=gpfs://gpfs1/steai-share:/mnt/shared-storage-user/steai-share \
	--volume /data:/data \
	--max-wait-duration=60m0s \
	--custom-resources brainpp.cn/fuse=1 \
	-- bash
