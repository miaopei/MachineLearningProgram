#########################################################################
# File Name: face_lmdb.sh
# Author: MiaoPei
# mail: miaopei@baicells.com
# Created Time: 2019年01月14日 星期一 14时17分43秒
#########################################################################
#!/bin/sh
# Create the face_48 lmdb input
# N.B. set the path to the face_48 train + val data dirs

EXAMPLE=/home/miaopei/workdir/github/DeepLearning/MachineLearningProgram/face_detect
DATA=/home/miaopei/workdir/github/DeepLearning/MachineLearningProgram/face_detect
TOOLS=/home/miaopei/workdir/spark/caffe/build/tools

TRAIN_DATA_ROOT=/home/miaopei/workdir/github/DeepLearning/MachineLearningProgram/face_detect/train
VAL_DATA_ROOT=/home/miaopei/workdir/github/DeepLearning/MachineLearningProgram/face_detect/val

# Set RESIZE=true to resize the image to 60x60. leave as false if image have
# already been resized using another tool.
RESIZE=true
if RESIZE; then
	RESIZE_HEIGHT=227
	RESIZE_WIDTH=337
else
	RESIZE_HEIGHT=0
	RESIZE_WIDTH=0
fi

if [ ! -d "${TRAIN_DATA_ROOT}" ]; then
	echo "Error: TRAIN_DATA_ROOT is not a path to a directory: ${TRAIN_DATA_ROOT}"
	echo "Set the TRAIN_DATA_ROOT variable in create_face_48.sh to the path" \
		 "where the face_48 training data is stored."
	exit 1
fi

if [ ! -d "${VAL_DATA_ROOT}" ]; then
	echo "Error: VAL_DATA_ROOT is not a path to a directory: ${VAL_DATA_ROOT}"
	echo "Set the VAL_DATA_ROOT variable in create_face_48.sh to the path" \
		 "where the face_48 training data is stored."
	exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 ${TOOLS}/convert_imageset \
	--resize_height=${RESIZE_HEIGHT} \
	--resize_width=${RESIZE_WIDTH} \
	--shuffle \
	${TRAIN_DATA_ROOT} \
	${DATA}/train.txt \
	${EXAMPLE}/face_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 ${TOOLS}/convert_imageset \
	--resize_height=${RESIZE_HEIGHT} \
	--resize_width=${RESIZE_WIDTH} \
	--shuffle \
	${VAL_DATA_ROOT} \
	${DATA}/val.txt \
	${EXAMPLE}/face_val_lmdb

echo "Done."

