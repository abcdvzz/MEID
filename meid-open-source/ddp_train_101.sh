# FEATURE_NAME='ResNet101'
FEATURE_NAME='ResNet50'
# FEATURE_NAME='TSM-R50'

# export CUDA_VISIBLE_DEVICES='4,5,6,7'
# export CUDA_VISIBLE_DEVICES='1'
python -m torch.distributed.launch --nproc_per_node 6 --master_port 29504 dy_main_ddp_101.py \
     --feature_name $FEATURE_NAME \
     --lr 0.001 \
     --gd 20 --lr_steps 30 60 --epochs 100 \
     --batch-size 256 -j 0 \
     --eval-freq 1 \
     --print-freq 20 \
     --root_log=$FEATURE_NAME-log \
     --root_model=$FEATURE_NAME'-checkpoints' \
     --store_name=$FEATURE_NAME'_v8_train_150_distill_0.1_bs256x6_lr0.001_step_30_60' \
     --gamma 1 \
     --num_class=1004 \
     --model_name=Twins_Pos_101 \
     --train_num_frames=150 \
     --val_num_frames=150 \
     --loss_func=FocalLoss \
     --clip_length=150 \
     --warm_epoch=0 \
     --ratio=0.5 