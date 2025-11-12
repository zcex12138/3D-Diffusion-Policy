# Examples:
# zsh scripts/train_policy.sh dp3 pick_and_place_point_cloud 1001
# zsh scripts/train_policy.sh fm pick_and_place_pc_and_pos 1020

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
exp_name=${task_name}-${addition_info}

export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=0

python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            exp_name=${exp_name} \
                            training.seed=0


