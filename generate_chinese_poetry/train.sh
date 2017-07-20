#!/bin/sh
set -x

save_dir="v1_version"

if [ -d $save_dir/models ];then
  rm -rf $save_dir/models
fi
mkdir -p $save_dir/models

config="v1_version/lstm_seq2seq.py"

export PYTHONPATH=$PWD:$PWD/v1_version/dataprovider/
echo $config
# export GLOG_v=2
# cgdb --args \
./bin/paddle_trainer \
--config=$config \
--log_period=5 \
--dot_period=50000 \
--test_period=5 \
--test_all_data_in_one_period=0 \
--saving_period=1 \
--save_dir=$save_dir/models \
--num_passes=100 \
--use_gpu=1 \
--trainer_count=4 \
--local=1 \
2>&1 | tee v1_train.log

