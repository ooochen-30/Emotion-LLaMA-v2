
#!/bin/bash
set -ex

__conda_setup="$('/home/user/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate emotion-llama-v2


ROOT_DIR="/path/to/the/infer/result"
mkdir -p "$ROOT_DIR"

LOGFILE="${ROOT_DIR}/score_log.txt"
: > "$LOGFILE"


NN_DATASETS=("cmu_mosi" "cmu_mosei" "sims" "ch_sims_v2_s")
MULTI_DATASETS=("mafw_multi" "bold")
SINGLE_DATASETS=("caer" "e3" "dfew" "mafw" "meld_sentiment" "mc_eiu_intent" "mc_eiu_emotion")

FILE_PATH="${ROOT_DIR}/MERUniBench-ht"
python evaluation/score_hit_ov.py --path "$FILE_PATH" >> "$LOGFILE" 2>&1

for dataset in "${NN_DATASETS[@]}"; do
    FILE_PATH="${ROOT_DIR}/MERUniBench-nn/${dataset}"
    echo "[INFO] Solve NN dataset: $dataset" >> "$LOGFILE"
    python evaluation/score_split_sentiment_no_neutral.py --file "$FILE_PATH" --split test >> "$LOGFILE" 2>&1
done

for dataset in "${MULTI_DATASETS[@]}"; do
    FILE_PATH="${ROOT_DIR}/Multi/${dataset}"
    echo "[INFO] Solve Multi dataset: $dataset" >> "$LOGFILE"
    python evaluation/score_multilabel.py --file "$FILE_PATH" >> "$LOGFILE" 2>&1
done

for dataset in "${SINGLE_DATASETS[@]}"; do
    FILE_PATH="${ROOT_DIR}/Single/${dataset}"
    echo "[INFO] Solve Single dataset: $dataset" >> "$LOGFILE"
    python evaluation/score_split.py --file "$FILE_PATH" --split test >> "$LOGFILE" 2>&1
done

