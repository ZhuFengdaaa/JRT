CUDA_VISIBLE_DEVICES=0 xvfb-run -s "-ac -screen 0 1280x1024x24" -a python main.py --env_type indoor --env_name roomgoal_mp3d_s --checkpoint_dir episode_mp3d_s $@
# CUDA_VISIBLE_DEVICES=$3 xvfb-run -s "-ac -screen $4 1280x1024x24" -a python evaluate.py --env_type indoor --env_name roomgoal_mp3d_sf --seed 0 --checkpoint_dir $1 --score_file $2 --use_pixel_change=False --use_value_replay=False --use_reward_prediction=False $@
# CUDA_VISIBLE_DEVICES=$3 xvfb-run -s "-ac -screen $4 1280x1024x24" -a python evaluate.py --env_type indoor --env_name roomgoal_mp3d_sf --seed 999 --checkpoint_dir $1 --score_file $2 --use_pixel_change=False --use_value_replay=False --use_reward_prediction=False $@
# CUDA_VISIBLE_DEVICES=$3 xvfb-run -s "-ac -screen $4 1280x1024x24" -a python evaluate.py --env_type indoor --env_name roomgoal_mp3d_sf --seed 2468 --checkpoint_dir $1 --score_file $2 --use_pixel_change=False --use_value_replay=False --use_reward_prediction=False $@
# CUDA_VISIBLE_DEVICES=$3 xvfb-run -s "-ac -screen $4 1280x1024x24" -a python evaluate.py --env_type indoor --env_name roomgoal_mp3d_sf --seed 3579 --checkpoint_dir $1 --score_file $2 --use_pixel_change=False --use_value_replay=False --use_reward_prediction=False $@
