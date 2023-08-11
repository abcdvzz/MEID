FEATURE_NAME='ResNet50'
# FEATURE_NAME='ResNet101'
# FEATURE_NAME='TSM-R50'

# export CUDA_VISIBLE_DEVICES='6,7' --master_port 29501
python -m torch.distributed.launch --nproc_per_node 6 dy_main_ddp.py \
     --feature_name $FEATURE_NAME \
     --resample 'CBS' \
     --lr 0.001 \
     --gd 20 --lr_steps 30 60 --epochs 100 \
     --batch-size 256 -j 0 \
     --eval-freq 1 \
     --print-freq 20 \
     --root_log=$FEATURE_NAME-log \
     --root_model=$FEATURE_NAME'-checkpoints' \
     --store_name=$FEATURE_NAME'_v10_bs256x6_lr0.001_step_30_60_test' \
     --gamma 1 \
     --num_class=1004 \
     --model_name=Twins_Pos \
     --train_num_frames=150 \
     --val_num_frames=150 \
     --loss_func=FocalLoss \
     --clip_length=150 \
     --warm_epoch=0 \
     --ratio=0.5 \
     --resume '/data/lxj/code/videolt/ResNet50-checkpoints/ResNet50_v10_bs256x6_lr0.001_step_30_60/ckpt.best.pth.tar'
