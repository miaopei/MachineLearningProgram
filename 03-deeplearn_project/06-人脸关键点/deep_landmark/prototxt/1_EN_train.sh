#!/bin/bash

#nohup 
    /home/matt/Documents/caffe/build/tools/caffe train \
    --solver=/home/admin01/workspace/deep_landmark/prototxt/1_EN_solver.prototxt \
    --gpu=0,1 \
    #--weights=/home/matt/Documents/caffe/faceLaMa/VGG_FACE.caffemodel \
    #>/home/admin01/workspace/deep_landmark/prototxt/log_1_F.log&
