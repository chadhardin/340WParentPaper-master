# DS 340W Research Paper on using Edge-Detect
Below is the original paper and citations for the work from the authors.
This code was modified multiple times to get updated and working correctly.

# How to run Edge-Detect
1) Use npy_generator.py to generate the training and testing split

2) Use testing/pi_queue.py 
  - Using the python terminal, use the code "python ./testing/pi_queue 100 128"
  - The first value is the size of each of the windows
  - The second value is the amount of neurons are in the first two layers of the model
  - Included in this repo is the saved model that got the scores put in the paper I submitted to class
  - After model is fitted and evaluated, the metrics will print on screen


# Edge-Detect (from original authors)
Repository for IEEE CCNC'21 paper titled "[Edge-Detect: Edge-centric Network Intrusion Detection using Deep Neural
Network](https://edas.info/showManuscript.php?type=stamped-e&m=1570662712&ext=pdf&title=PDF+file)"
# Citation
Please cite if you find our work useful

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

Following 3 options are available for any clarification, comments or suggestions
- Join the [discussion forum](https://github.com/racsa-lab/Edge-Detect/discussions).
- Create an [issue](https://github.com/racsa-lab/Edge-Detect/issues).
- Contact the authors.
