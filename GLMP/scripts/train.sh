#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

HOMEDIR=${HOME}
BASEDIR="${HOMEDIR}/myGLMP"
train="${BASEDIR}/GLMP/train.py"

BS=8
ACCUM=1
EPOCH=50
LSTEP=10
ML=6
MH=3
SL=30
NAME=GLMP
LR=1e-3
TRAINFILE=train.txt
DEVFILE=dev.txt
SEED=2021

python -u $train \
	--train_file $TRAINFILE	\
	--dev_file $DEVFILE	\
	--model_name	$NAME	\
	--train_batch_size	$BS	\
	--learning_rate $LR	\
	--grad_accum_steps	$ACCUM	\
	--memory_len $ML\
	--memory_hop $MH \
	--sent_len $SL \
	--n_epochs	$EPOCH	\
	--logging_steps $LSTEP \
	--do_train	\
	--do_eval	\
	--save_checkpoints