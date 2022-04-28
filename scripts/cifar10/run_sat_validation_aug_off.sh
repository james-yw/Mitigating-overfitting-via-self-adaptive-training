DATASET=cifar10
DATA_ROOT='~/datasets/CIFAR10'
ARCH=resnet34
LR=0.1
LR_SCHEDULE='cosine'
EPOCHS=200
BATCH_SIZE=256
LOSS=sat
ALPHA=0.7
ES=25
NOISE_RATE=0.4
NOISE_TYPE='corrupted_label'
TRAIN_SETS='train'
VAL_SETS='clean_train noisy_train clean_val noisy_val'
EXP_NAME=${DATASET}/validation_aug_off_${ARCH}_${LOSS}_${NOISE_TYPE}_r${NOISE_RATE}_m${ALPHA}_p${ES}_${LR_SCHEDULE}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='0'

### print info
echo ${EXP_NAME}
mkdir -p ckpts
mkdir -p logs

### train
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -u main.py --arch ${ARCH} --loss ${LOSS} \
        --sat-alpha ${ALPHA} --sat-es ${ES} \
        --dataset ${DATASET} --data-root ${DATA_ROOT} --turn-off-aug \
        --noise-rate ${NOISE_RATE} --noise-type ${NOISE_TYPE} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
        --train-sets ${TRAIN_SETS} --val-sets ${VAL_SETS} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        >> ${LOG_FILE} 2>&1
