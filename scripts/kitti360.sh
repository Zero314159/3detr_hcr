CUDA_VISIBLE_DEVICES=2,3 \
python main.py \
--dataset_name kitti360 \
--nqueries 256 \
--max_epoch 10000 \
--matcher_giou_cost 2 \
--matcher_cls_cost 1 \
--matcher_center_cost 0 \
--matcher_objectness_cost 0 \
--loss_giou_weight 1 \
--loss_no_object_weight 0.25 \
--eval_every_epoch 10 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir outputs/kitti360_test \
--base_lr 1e-4