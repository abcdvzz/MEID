FEATURE_NAME='Charades'

# export CUDA_VISIBLE_DEVICES='0,1,2,3'
python -m torch.distributed.launch --nproc_per_node 2 --master_port 29503 dy_main_ddp_diff.py \
     --feature_name $FEATURE_NAME \
     --resample 'CBS' \
     --lr 0.0001 \
     --gd 20 --lr_steps 30 60 --epochs 100 \
     --batch-size 128 -j 0 \
     --eval-freq 5 \
     --print-freq 20 \
     --root_log=$FEATURE_NAME-log \
     --root_model=$FEATURE_NAME'-checkpoints' \
     --store_name=$FEATURE_NAME'_v11_r101_bs128x2_lr0.0001_step_30_60_test' \
     --gamma 1 \
     --num_class=157 \
     --model_name=Twins_Pos \
     --train_num_frames=200 \
     --val_num_frames=200 \
     --loss_func=FocalLoss \
     --clip_length=200 \
     --warm_epoch=0 \
     --ratio=0.5 \
     --head=100 \
     --tail=20 \
     --pretrain '/data/lxj/code/videolt/Charades-checkpoints/Charades_v9_r101_bs128x2_lr0.0001_step_30_60/ckpt.best.pth.tar'
