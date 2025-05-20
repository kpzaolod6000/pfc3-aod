python dehaze_one_metrics.py --model_path /home/pytorch/data/snapshots/NYU/L1/dehazer_final.pth --hazy_image_path /home/pytorch/data/results_coco/heavy_haze/000000000139.jpg
 --orig_image_path /home/pytorch/data/coco_val2017/val2017/000000000139.jpg

python dehaze_one_metrics_paod.py --model_path /home/pytorch/data/snapshots/NYU/PAOD_MS_SSIM_L2/dehazer_final.pth --hazy_image_path /home/pytorch/data/data/data/NYU2_1_7_3.jpg --orig_image_path /home/pytorch/data/data/images/NYU2_1.jpg

## python commands coco
python --model_path /home/pytorch/data/snapshots/NYU/L1/dehazer_final.pth --hazy_folder_path /home/pytorch/data/results_coco/heavy_haze --gt_folder_path /home/pytorch/data/coco_val2017/val2017

## validation NYU
python val_nyu_aod_colection.py --model_path /home/pytorch/data/snapshots/NYU/MS_SSIM_L1/dehazer_final.pth --hazy_folder_path /home/pytorch/data/results_coco/medium_haze/ --gt_folder_path /home/pytorch/data/coco_val2017/val2017

python val_nyu_paod_colection.py --model_path /home/pytorch/data/snapshots/NYU/PAOD_MS_SSIM_L2_G/dehazer_final.pth --hazy_folder_path /home/pytorch/data/results_gun/val2024/heavy_haze/ --gt_folder_path /home/pytorch/data/images_gun/val2024/

## DEHAZE FOLDER
python batch_dehaze.py \
    --haze_dir /ruta/a/imagenes_haze \
    --output_dir /ruta/a/imagenes_dehaze \
    --weight_path /ruta/a/dehazer_final.pth
