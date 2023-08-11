FEATURE_NAME='Charades'

export CUDA_VISIBLE_DEVICES='6,7'
python -m torch.distributed.launch --nproc_per_node 2 --master_port 29503 dy_main_ddp_101.py \
     --feature_name $FEATURE_NAME \
     --lr 0.0001 \
     --gd 20 --lr_steps 30 60 --epochs 100 \
     --batch-size 128 -j 0 \
     --eval-freq 1 \
     --print-freq 20 \
     --root_log=$FEATURE_NAME-log \
     --root_model=$FEATURE_NAME'-checkpoints' \
     --store_name=$FEATURE_NAME'_v9_r101_bs128x2_lr0.0001_step_30_60' \
     --gamma 1 \
     --num_class=157 \
     --model_name=Twins_Pos_101 \
     --train_num_frames=200 \
     --val_num_frames=200 \
     --loss_func=FocalLoss \
     --clip_length=200 \
     --warm_epoch=0 \
     --ratio=0.5 \
     --head=100 \
     --tail=20 
