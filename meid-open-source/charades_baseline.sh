FEATURE_NAME='Charades'

export CUDA_VISIBLE_DEVICES='9'
python -m torch.distributed.launch --nproc_per_node 1 --master_port 29504 base_main_ddp.py  \
     --feature_name $FEATURE_NAME \
     --lr 0.0001 \
     --gd 20 --lr_steps 30 60 --epochs 100 \
     --batch-size 1024 -j 16 \
     --eval-freq 5 \
	--print-freq 20 \
     --root_log=$FEATURE_NAME-log \
     --root_model=$FEATURE_NAME'-checkpoints' \
     --store_name=$FEATURE_NAME'_r101_bs1024_lr0.0001_nextvlad_EQL' \
     --num_class=157 \
     --model_name=NeXtVLADModel \
     --train_num_frames=200 \
     --val_num_frames=200 \
     --loss_func=EQL \
     --head=100 \
     --tail=20 
