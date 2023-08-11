FEATURE_NAME='ResNet50'

export CUDA_VISIBLE_DEVICES='0'
python -m torch.distributed.launch --nproc_per_node 1 dy_main_ddp_ori.py \
     --feature_name $FEATURE_NAME \
     --lr 0.0001 \
     --gd 20 --lr_steps 30 60 --epochs 100 \
     --batch-size 512 -j 0 \
     --eval-freq 1 \
     --print-freq 20 \
     --root_log=$FEATURE_NAME-log \
     --root_model=$FEATURE_NAME'-checkpoints' \
     --store_name=$FEATURE_NAME'_r50_bs512_lr0.0001_lateavg_framestack_warm0_ratio0.5_Focalloss_gamma1' \
     --gamma 1 \
     --num_class=1004 \
     --model_name=NonlinearClassifier \
     --train_num_frames=60 \
     --val_num_frames=150 \
     --loss_func=FocalLoss \
     --clip_length=60 \
     --warm_epoch=0 \
     --ratio=0.5 