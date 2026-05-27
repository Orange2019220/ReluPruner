

#
#
#python train_with_taylor_pruning_global_relu_pruning_abs.py --path_t='save/models/ResNet18_cifar10_batch128/ResNet18_best.pth' --alpha=0.5 --gamma=0.5 --model_s='CustomResNet18' --dataset='cifar10' --distill='kd' --global_keep_ratio=0.1
#
#python train_with_taylor_pruning_global_relu_pruning_abs.py --path_t='save/models/ResNet18_cifar10_batch128/ResNet18_best.pth' --alpha=0.5 --gamma=0.5 --model_s='CustomResNet18' --dataset='cifar10' --distill='kd' --global_keep_ratio=0.026
#
#python train_with_taylor_pruning_global_relu_pruning_abs.py --path_t='save/models/ResNet18_cifar10_batch128/ResNet18_best.pth' --alpha=0.5 --gamma=0.5 --model_s='CustomResNet18' --dataset='cifar10' --distill='kd' --global_keep_ratio=0.013
#
#python train_with_taylor_pruning_global_relu_pruning_abs.py --path_t='save/models/ResNet18_cifar10_batch128/ResNet18_best.pth' --alpha=0.5 --gamma=0.5 --model_s='CustomResNet18' --dataset='cifar10' --distill='kd' --global_keep_ratio=0.6
#
#python train_with_taylor_pruning_global_relu_pruning_abs.py --path_t='save/models/ResNet18_cifar10_batch128/ResNet18_best.pth' --alpha=0.5 --gamma=0.5 --model_s='CustomResNet18' --dataset='cifar10' --distill='kd' --global_keep_ratio=0.51
#
#python train_with_taylor_pruning_global_relu_pruning_abs.py --path_t='save/models/ResNet18_cifar10_batch128/ResNet18_best.pth' --alpha=0.5 --gamma=0.5 --model_s='CustomResNet18' --dataset='cifar10' --distill='kd' --global_keep_ratio=0.3

# 94.02
python train_student_stage2_multigpu.py --dataset='cifar10' --path_t='save/models/ResNet18_cifar10_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_cifar10_0.1_before_abs_kdT4_batch128/CustomResNet18_best_at_budget.pth'
# 90.1200
python train_student_stage2_multigpu.py --dataset='cifar10' --path_t='save/models/ResNet18_cifar10_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_cifar10_0.026_before_abs_kdT4_batch128/CustomResNet18_best_at_budget.pth'
# 87.82
python train_student_stage2_multigpu.py --dataset='cifar10' --path_t='save/models/ResNet18_cifar10_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_cifar10_0.013_before_abs_kdT4_batch128/CustomResNet18_best_at_budget.pth'


# 95.180
python train_student_stage2_multigpu.py --dataset='cifar10' --path_t='save/models/ResNet18_cifar10_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_cifar10_0.6_before_abs_kdT4_batch128/CustomResNet18_best_at_budget.pth'
#
python train_student_stage2_multigpu.py --dataset='cifar10' --path_t='save/models/ResNet18_cifar10_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_cifar10_0.51_before_abs_kdT4_batch128/CustomResNet18_best_at_budget.pth'


python train_student_stage2_multigpu.py --dataset='cifar10' --path_t='save/models/ResNet18_cifar10_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_cifar10_0.3_before_abs_kdT4_batch128/CustomResNet18_best_at_budget.pth'


python train_student_stage2_multigpu.py --dataset='cifar100' --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_cifar100_0.026_before_abs_kdT4_batch128/CustomResNet18_best_at_budget.pth'
python train_student_stage2_multigpu.py --dataset='cifar100' --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_cifar100_0.013_before_abs_kdT4_batch128/CustomResNet18_best_at_budget.pth'
python train_student_stage2_multigpu.py --dataset='cifar100' --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_cifar100_0.3_before_abs_kdT4_batch128/CustomResNet18_best_at_budget.pth'
python train_student_stage2_multigpu.py --dataset='cifar100' --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_cifar100_0.37_before_abs_kdT4_batch128/CustomResNet18_best_at_budget.pth'
python train_student_stage2_multigpu.py --dataset='cifar100' --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_cifar100_0.4_before_abs_kdT4_batch128/CustomResNet18_best_at_budget.pth'
python train_student_stage2_multigpu.py --dataset='cifar100' --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dense --distill='attention' --learning_rate=0.01 --path_s_load='save/student_model/stage1/S_CustomResNet18_T1_ResNet18_cifar100_0.6_before_abs_kdT4_batch128/CustomResNet18_best_at_budget.pth'



# senet

python train_student_stage1_multigpu.py --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dataset='cifar100' --distill='kd' --sensitivity='ResNet18_c100_relu82k_sensitivity'

python train_student_stage1_multigpu.py --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dataset='cifar100' --distill='kd' --sensitivity='ResNet18_c100_relu50k_sensitivity'
python train_student_stage1_multigpu.py --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dataset='cifar100' --distill='kd' --sensitivity='ResNet18_c100_relu25k_sensitivity'
python train_student_stage1_multigpu.py --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dataset='cifar100' --distill='kd' --sensitivity='ResNet18_c100_relu100k_sensitivity'
python train_student_stage1_multigpu.py --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' --gamma=0.5 --alpha=0.5 --beta=1000 --model_s='CustomResNet18' --dataset='cifar100' --distill='kd' --sensitivity='ResNet18_c100_relu120k_sensitivity'
