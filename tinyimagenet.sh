



python train_with_taylor_pruning_global_relu_pruning_abs.py --path_t='save/models/ResNet18_tiny_imagenet/ResNet18_best.pth.tar' --alpha=0.5 --gamma=0.5 --model_s='CustomResNet18' --dataset='tiny_imagenet' --distill='kd' --global_keep_ratio=0.12

python train_with_taylor_pruning_global_relu_pruning_abs.py --path_t='save/models/ResNet18_tiny_imagenet_batch32/ResNet18_best.pth' --alpha=0.5 --gamma=0.5 --model_s='CustomResNet18' --dataset='tiny_imagenet' --distill='kd' --global_keep_ratio=0.06

python train_with_taylor_pruning_global_relu_pruning_abs.py --path_t='save/models/ResNet18_tiny_imagenet_batch32/ResNet18_best.pth' --alpha=0.5 --gamma=0.5 --model_s='CustomResNet18' --dataset='tiny_imagenet' --distill='kd' --global_keep_ratio=0.03

python train_with_taylor_pruning_global_relu_pruning_abs.py --path_t='save/models/ResNet18_tiny_imagenet_batch32/ResNet18_best.pth' --alpha=0.5 --gamma=0.5 --model_s='CustomResNet18' --dataset='tiny_imagenet' --distill='kd' --global_keep_ratio=0.3

python train_with_taylor_pruning_global_relu_pruning_abs.py --path_t='save/models/ResNet18_tiny_imagenet_batch32/ResNet18_best.pth' --alpha=0.5 --gamma=0.5 --model_s='CustomResNet18' --dataset='tiny_imagenet' --distill='kd' --global_keep_ratio=0.4

python train_with_taylor_pruning_global_relu_pruning_abs.py --path_t='save/models/ResNet18_tiny_imagenet_batch32/ResNet18_best.pth' --alpha=0.5 --gamma=0.5 --model_s='CustomResNet18' --dataset='tiny_imagenet' --distill='kd' --global_keep_ratio=0.6


python train_student_stage2_multigpu.py --dataset='tiny_imagenet' --path_t='save/models/ResNet18_tiny_imagenet_batch32/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_tiny_imagenet_0.12_before_abs_kdT4_batch32/CustomResNet18_best_at_budget.pth'
python train_student_stage2_multigpu.py --dataset='tiny_imagenet' --path_t='save/models/ResNet18_tiny_imagenet_batch32/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_tiny_imagenet_0.06_before_abs_kdT4_batch32/CustomResNet18_best_at_budget.pth'
python train_student_stage2_multigpu.py --dataset='tiny_imagenet' --path_t='save/models/ResNet18_tiny_imagenet_batch32/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_tiny_imagenet_0.03_before_abs_kdT4_batch32/CustomResNet18_best_at_budget.pth'
python train_student_stage2_multigpu.py --dataset='tiny_imagenet' --path_t='save/models/ResNet18_tiny_imagenet_batch32/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_tiny_imagenet_0.3_before_abs_kdT4_batch32/CustomResNet18_best_at_budget.pth'
python train_student_stage2_multigpu.py --dataset='tiny_imagenet' --path_t='save/models/ResNet18_tiny_imagenet_batch32/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_tiny_imagenet_0.4_before_abs_kdT4_batch32/CustomResNet18_best_at_budget.pth'
python train_student_stage2_multigpu.py --dataset='tiny_imagenet' --path_t='save/models/ResNet18_tiny_imagenet_batch32/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_tiny_imagenet_0.6_before_abs_kdT4_batch32/CustomResNet18_best_at_budget.pth'

