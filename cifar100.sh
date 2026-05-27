

nohup python train_with_taylor_pruning_global_relu_pruning_abs.py --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' --alpha=0.5 --gamma=0.5 --model_s='CustomResNet18' --dataset='cifar100' --distill='kd' --global_keep_ratio=0.37
nohup python train_with_taylor_pruning_global_relu_pruning_abs.py --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' --alpha=0.5 --gamma=0.5 --model_s='CustomResNet18' --dataset='cifar100' --distill='kd' --global_keep_ratio=0.4
nohup python train_with_taylor_pruning_global_relu_pruning_abs.py --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' --alpha=0.5 --gamma=0.5 --model_s='CustomResNet18' --dataset='cifar100' --distill='kd' --global_keep_ratio=0.6

nohup python train_teacher.py --model='ResNet18' --dataset='tiny_imagenet' --batch_size 32 --epochs 100 --learning_rate 0.05 --momentum 0.9

nohup python train_student_stage2_multigpu.py --dataset='cifar100' --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_cifar100_0.26_before_abs_kdT4_batch128/CustomResNet18_best_at_budget.pth'
nohup python train_student_stage2_multigpu.py --dataset='cifar100' --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_cifar100_0.13_before_abs_kdT4_batch128/CustomResNet18_best_at_budget.pth'
