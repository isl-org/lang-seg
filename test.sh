export CUDA_VISIBLE_DEVICES=0; python test_lseg.py --backbone clip_vitl16_384 --eval --dataset ade20k --data-path ../datasets/ \
--weights checkpoints/lseg_ade20k_l16.ckpt --widehead --no-scaleinv 




