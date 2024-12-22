 python3 haze_video_coco.py
 python tunning.py --loss_type ms-ssim+l2
 python tunning.py --loss_type ms-ssim+l1
 python tunning.py --loss_type l1
 python tunning.py --loss_type l2

 python tunning_paod.py --loss_type ms-ssim
 python tunning_paod.py --loss_type ms-ssim+l1
 python tunning_paod.py --loss_type ms-ssim+l2
 python tunning_paod.py --loss_type l1
 python tunning_paod.py --loss_type l2


 python train_nyu.py --loss_type ms-ssim+l2
 python train_nyu_paod.py --loss_type ms-ssim+l2