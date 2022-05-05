DATASET=SVHN
DATA_ROOT='../DATASETS/SVHN'
ARCH=resnet18
LR=0.1
LR_SCHEDULE='cosine'
EPOCHS=100
BATCH_SIZE=128
LOSS=ce
NOISE_RATE=0.4

#NOISE_TYPE='corrupted_label'
NOISE_TYPE='Gaussian'
#NOISE_TYPE='random_pixels'
#NOISE_TYPE='shuffled_pixels'


TRAIN_SETS='trainval'
VAL_SETS='test_set'
EXP_NAME=${DATASET}/${ARCH}_${LOSS}_${NOISE_TYPE}_r${NOISE_RATE}_${LR_SCHEDULE}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='0'
RESUME='./ckpts/SVHN/resnet18_ce_corrupted_label_r0_cosine_/checkpoint_latest.tar'
RESULT_DIR=results/${EXP_NAME}

### print info
echo ${EXP_NAME}
mkdir -p ckpts/${DATASET}
mkdir -p logs/${DATASET}
mkdir -p results/${DATASET}

### train
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -u main.py --arch ${ARCH} --loss ${LOSS} \
        --dataset ${DATASET} --data-root ${DATA_ROOT} \
        --noise-rate ${NOISE_RATE} --noise-type ${NOISE_TYPE} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
        --train-sets ${TRAIN_SETS} --val-sets ${VAL_SETS} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        --result-dir ${RESULT_DIR} \
        >> ${LOG_FILE} 2>&1
