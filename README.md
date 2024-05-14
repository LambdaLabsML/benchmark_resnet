# benchmark_resnet

```
ray job submit --address="http://ip-dashboard:8265" \
--entrypoint-num-gpus 1 \
--runtime-env-json='{"working_dir": ".", "pip": ["torch", "torchvision"]}' \
-- python benchmark_resnet_imagenet_ray.py \
--batch-size 256 \
--epochs 2 \
--repeat 1 \
--num-workers 16 \
--pin-memory \
--persistent-workers \
--prefetch-factor 2 \
--storage-path /mnt/cluster_storage \
--dataset-path /mnt/cluster_storage/tiny-224 \
--num-gpu-workers 32
```
