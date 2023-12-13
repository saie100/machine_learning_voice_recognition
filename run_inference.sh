#!/bin/bash

# Loop 50 times
# for i in {1..50}
# do
#    echo "Run #$i relu"
#    python3 inference.py relu models/machine_learning_model_relu$i.pth
# done

# for i in {1..50}
# do
#    echo "Run #$i tanh"
#    python3 inference.py tanh models/machine_learning_model_tanh$i.pth
# done

for i in {1..50}
do
   echo "Run #$i sigmoid"
   python3 inference.py sigmoid models/machine_learning_model_sigmoid$i.pth
done
