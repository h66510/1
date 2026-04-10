 rlaunch --cpu=32 --memory=102400 \
 	 --mount=gpfs://gpfs1/gaoao-p:/mnt/shared-storage-user/gaoao-p \
	 --mount=gpfs://gpfs1/tanxin:/mnt/shared-storage-user/tanxin \
	 --mount=gpfs://gpfs1/steai-share:/mnt/shared-storage-user/steai-share \
	 --custom-resources brainpp.cn/fuse=1 \
	 --charged-group=steai_cpu_task \
       	 --max-wait-duration=60m0s \
	 --custom-resources brainpp.cn/fuse=1 \
	 -- bash