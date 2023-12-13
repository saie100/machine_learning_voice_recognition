#!/bin/bash

# Loop 50 times
# for i in {1..50}
# do
#    echo "Run #$i"
#    python3 train_model.py relu models/machine_learning_model_relu$i.pth
# done

# Loop 50 times
for i in {1..50}
do
   echo "Run #$i"
   python3 train_model.py tanh models/machine_learning_model_tanh$i.pth
done

# Loop 50 times
for i in {1..50}
do
   echo "Run #$i"
   python3 train_model.py sigmoid models/machine_learning_model_sigmoid$i.pth
done
