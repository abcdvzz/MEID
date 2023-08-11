FEATURE_NAME='ResNet50'
# FEATURE_NAME='ResNet101'
# FEATURE_NAME='TSM-R50'

# export CUDA_VISIBLE_DEVICES='4,5,6,7'
python -m torch.distributed.launch --nproc_per_node 8 --master_port 29504 dy_main_ddp_diff.py \
     --feature_name $FEATURE_NAME \
     --resample 'CBS' \
     --lr 0.001 \
     --gd 20 --lr_steps 30 60 --epochs 100 \
     --batch-size 128 -j 0 \
     --eval-freq 1 \
     --print-freq 20 \
     --root_log=$FEATURE_NAME-log \
     --root_model=$FEATURE_NAME'-checkpoints' \
     --store_name=$FEATURE_NAME'_v11_bs128x8_diff_1_lr0.001_step_30_60' \
     --gamma 1 \
     --num_class=1004 \
     --model_name=Twins_Pos_diff \
     --train_num_frames=150 \
     --val_num_frames=150 \
     --loss_func=FocalLoss \
     --clip_length=150 \
     --warm_epoch=0 \
     --ratio=0.5 \
     --pretrain '/data/lxj/code/videolt/ResNet50-checkpoints/ResNet50_v9_bs256x6_lr0.001_step_30_60/ckpt.best.pth.tar'
