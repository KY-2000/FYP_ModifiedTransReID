# train
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 66666 train.py --config_file /home/cjy/data0/TransReID/configs_ViT/OCC_Duke/vit_transreid_frm05.yml MODEL.DIST_TRAIN True
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 66666 train.py --config_file configs/VeRi/vit_transreid_stride.yml MODEL.DIST_TRAIN True
# test
# python test.py --config_file configs/DukeMTMC/vit_transreid.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"
# python test.py --config_file /home/cjy/data0/TransReID/configs_ViT/DukeMTMC/vit_transreid.yml MODEL.DEVICE_ID "('0')"  TEST.WEIGHT /home/cjy/data0/TransReID/logs_ViT/dukemtmc/vit_transreid_pkr1.0_frm0.3_before_global_method1/transformer_120.pth
