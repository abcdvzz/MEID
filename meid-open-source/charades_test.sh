FEATURE_NAME='Charades'
CKPT='/data/lxj/code/videolt/Charades-checkpoints/Charades_r101_bs1024_lr0.0001_nextvlad_EQL/ckpt.best.pth.tar'

export CUDA_VISIBLE_DEVICES='0'
python test_ddp.py \
     --feature_name $FEATURE_NAME \
     --resume $CKPT \
     --batch-size 7 -j 0 \
     --eval-freq 5 \
     --print-freq 20 \
     --gamma 1 \
     --num_class=157 \
     --model_name=NeXtVLADModel \
     --train_num_frames=200 \
     --val_num_frames=200 \
     --loss_func=EQL \
     --clip_length=200 \
     --warm_epoch=0 \
     --ratio=0.5 \
     --head=100 \
     --tail=20 
