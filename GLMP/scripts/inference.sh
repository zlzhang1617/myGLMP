#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

HOMEDIR=${HOME}
BASEDIR="${HOMEDIR}/myGLMP"
train="${BASEDIR}/GLMP/train.py"

DEVFILE=test.txt
SEED=2021
MODEL=$1

if ["$MODEL" -eq ""]
then
	MODEL=GLMP_best
	echo "no model input. use default `GLMP_best`"
else
	echo "use $MODEL"
fi

python -u $train \
	--seed $SEED \
	--dev_file $DEVFILE	\
	--init_model_dir $MODEL