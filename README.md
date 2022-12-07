# DS 340W Research Paper on using Edge-Detect
This repository is based off the Edge-Detect research paper code. Since their publication, many updates had to be done to the code. This code is working (Dec, 2022) and from the testing done it has a Model.Evalute Accuracy of 94.33% using the tensorflow package. Sklearn also has functions for calculating model measurements, which resulted in an Accuracy of 82.94%, Precision of 89.95%, Recall of 93.89%, Kappa score of 79.44%, and an F1_score of 88.07. The original paper includes using this on a Raspberry Pi, but from the testing I've done I do not think running on a Raspberry Pi is feasible.

# How to run Edge-Detect
* The creation of these numpy files requires 32GB of ram, it will take ~30 min to run depending on the computer, and take up ~15GB of storage space
1) Use npy_generator.py to generate the training and testing split
   - This takes a while to complete, so numbers are printed throughout the process to keep track

* The total training and testing time will take ~1 hour
2) Use testing/pi_queue.py 
  - Using the python terminal, use the code "python ./testing/pi_queue 100 128"
  - The first value is the size of each of the windows
  - The second value is the amount of neurons are in the first two layers of the model
  - Included in this repo is the saved model that got the scores put in the paper I submitted to class
            - * I was having issues reloading this model, each time loaded has reduced the performance in my testing
  - After model is fitted and evaluated, the metrics will print on screen


# Citation
```bibtex
@INPROCEEDINGS{9369469,  
author={P. {Singh} and J. J. {P} and A. {Pankaj} and R. {Mitra}},  
booktitle={2021 IEEE 18th Annual Consumer Communications   Networking Conference (CCNC)},   
title={Edge-Detect: Edge-Centric Network Intrusion Detection using Deep Neural Network},   
year={2021},  
volume={},  
number={},  
pages={1-6},  
doi={10.1109/CCNC49032.2021.9369469}}
```
