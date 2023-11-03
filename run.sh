CUDA_VISIBLE_DEVICES=3  python3 deepsc_main.py  --cr  3  --batch_size 64 \
--test_batch_size 50  --lr 0.0001  --output_dir  output_cr3_vitsc_rt_8dB --model ViTSC \
 --resume  ckpt_response/ckpt-t-cr3-vitsc-t8db-awgn.pth