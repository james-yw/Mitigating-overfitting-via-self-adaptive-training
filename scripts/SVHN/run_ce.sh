DATASET=SVHN
DATA_ROOT='../DATASETS/SVHN'
ARCH=resnet18
LR=0.1
LR_SCHEDULE='cosine'
EPOCHS=100
BATCH_SIZE=128
LOSS=ce
NOISE_RATE=0
NOISE_TYPE='corrupted_label'
TRAIN_SETS='trainval'
VAL_SETS='test_set'
EXP_NAME=${DATASET}/${ARCH}_${LOSS}_${NOISE_TYPE}_r${NOISE_RATE}_${LR_SCHEDULE}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='0'
RESUME='./ckpts/SVHN/resnet18_ce_corrupted_label_r0_cosine_/checkpoint_latest.tar'

### print info
echo ${EXP_NAME}
mkdir -p ckpts/${DATASET}
mkdir -p logs/${DATASET}

### train
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -u main.py --arch ${ARCH} --loss ${LOSS} \
        --dataset ${DATASET} --data-root ${DATA_ROOT} \
        --noise-rate ${NOISE_RATE} --noise-type ${NOISE_TYPE} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
        --train-sets ${TRAIN_SETS} --val-sets ${VAL_SETS} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        --resume ${RESUME} \
        >> ${LOG_FILE} 2>&1
