FEATURE_NAME='ResNet50'
CKPT='ResNet50-checkpoints/ResNet50_v6_pos_moe_2_KL_bs1024_lr0.0001_step_30_60/ckpt.best.pth.tar'

export CUDA_VISIBLE_DEVICES='0'
python test_ddp.py \
     --feature_name $FEATURE_NAME \
     --resume $CKPT \
     --batch-size 7 -j 0 \
     --eval-freq 5 \
     --print-freq 20 \
     --gamma 1 \
     --num_class=1004 \
     --model_name=Twins_Pos \
     --train_num_frames=60 \
     --val_num_frames=150 \
     --loss_func=FocalLoss \
     --clip_length=60 \
     --warm_epoch=0 \
     --ratio=0.5 