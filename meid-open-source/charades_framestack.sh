FEATURE_NAME='Charades'

export CUDA_VISIBLE_DEVICES='8'
python -m torch.distributed.launch --nproc_per_node 1 --master_port 29505 dy_main_ddp_ori.py \
     --feature_name $FEATURE_NAME \
     --lr 0.0001 \
     --gd 20 --lr_steps 30 60 --epochs 100 \
     --batch-size 1024 -j 0 \
     --eval-freq 5 \
     --print-freq 20 \
     --root_log=$FEATURE_NAME-log \
     --root_model=$FEATURE_NAME'-checkpoints' \
     --store_name=$FEATURE_NAME'_r101_bs1024_lr0.0001_lateavg_framestack_warm0_ratio0.5_Focalloss_gamma1' \
     --gamma 1 \
     --num_class=157 \
     --model_name=NeXtVLADModel \
     --train_num_frames=200 \
     --val_num_frames=200 \
     --loss_func=FocalLoss \
     --clip_length=200 \
     --warm_epoch=0 \
     --ratio=0.5 \
     --head=100 \
     --tail=20 
