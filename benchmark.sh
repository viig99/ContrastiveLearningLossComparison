#!/usr/bin/env bash

# Run the benchmark
python train.py --loss_func info_nce 2>&1 | tee info_nce.log
python train.py --loss_func nt_xent 2>&1 | tee nt_xent.log
python train.py --loss_func dhel 2>&1 | tee dhel.log
python train.py --loss_func dcl 2>&1 | tee dcl.log
python train.py --loss_func vicreg 2>&1 | tee vicreg.log
python train.py --loss_func dcl_symmetric 2>&1 | tee dcl_symmetric.log
