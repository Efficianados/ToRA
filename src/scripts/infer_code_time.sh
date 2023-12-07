set -ex

# MODEL_NAME_OR_PATH="llm-agents/tora-code-34b-v1.0"
# MODEL_NAME_OR_PATH="llm-agents/tora-70b-v1.0"
MODEL_NAME_OR_PATH="llm-agents/tora-7b-v1.0"
# MODEL_NAME_OR_PATH=$HF_HOME/hub/models--llm-agents--tora-7b-v1.0/snapshots/717edbee98945192b1a396fc9c337c5b32d6c79c/

# DATA_LIST = ['math', 'gsm8k', 'gsm-hard', 'svamp', 'tabmwp', 'asdiv', 'mawps']

# DATA_NAME="math"
DATA_NAME="gsm8k"

SPLIT="test"
PROMPT_TYPE="tora"
# PROMPT_TYPE="pal"
NUM_TEST_SAMPLE=-1


CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python -um infer.inference_measure_tool_time \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--data_name ${DATA_NAME} \
--split ${SPLIT} \
--prompt_type ${PROMPT_TYPE} \
--use_train_prompt_format \
--num_test_sample ${NUM_TEST_SAMPLE} \
--seed 0 \
--temperature 0 \
--n_sampling 1 \
--top_p 1 \
--start 0 \
--end -1 \
