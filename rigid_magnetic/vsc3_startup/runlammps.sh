#!/bin/bash
N_rand1=$RANDOM
N_rand2=$RANDOM
N_rand3=$RANDOM
N_rand4=$RANDOM
N_rand5=$RANDOM

lmp -in in.mag2patch-quasi-2d -var seed1 $N_rand1 -var seed2 $N_rand2 -var seed3 $N_rand3 -var seed4 $N_rand4 -var temp Temperature -var seed5 $N_rand4 
