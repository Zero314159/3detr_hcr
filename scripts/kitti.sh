python -m torch.distributed.launch --nproc_per_node=2 \

CUDA_VISIBLE_DEVICES=0,1 \
python main.py \
--dataset_name kitti \
--nqueries 256 \
--max_epoch 1080 \
--matcher_giou_cost 2 \
--matcher_cls_cost 1 \
--matcher_center_cost 0 \
--matcher_objectness_cost 0 \
--loss_giou_weight 1 \
--loss_no_object_weight 0.25 \
--eval_every_epoch 10 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir outputs/kitti_1080 
