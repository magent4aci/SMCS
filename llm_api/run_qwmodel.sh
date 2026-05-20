# export CUDA_VISIBLE_DEVICES=0
# nohup python Qwen-7B-Chat/api.py --port 6001 >Qwen-7B-Chat/logs/qwen-7b-chat.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python Qwen1.5-110B/api.py --port 6001 >Qwen1.5-110B/logs/qwen-1.5-110b.log 2>&1 &

#nohup python your_path_to_api/api.py --port 6002 > log/Llama-3-70B-Instruct.log 2>&1 &
#
#export CUDA_VISIBLE_DEVICES=0
#nohup python your_path_to_api/api.py --port 6002 > log/qwen-1.5-7b-chat.log 2>&1 &
#
#export CUDA_VISIBLE_DEVICES=0
#nohup python your_path_to_api/api.py --port 6003 > log/qwen-2.7-instruct.log 2>&1 &
#
#export CUDA_VISIBLE_DEVICES=0
#nohup python your_path_to_api/api.py --port 6004 > log/qwen-2.5-instruct.log 2>&1 &

model path:
your path to Qwen2.5-Math-7B-Instruct
your path to Meta-Llama-3-70B-Instruct


for i in $(seq 1 20); do
  python test_api.py > log${i}.log &
done