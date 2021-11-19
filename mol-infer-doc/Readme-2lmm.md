---
title: "Readme for the 2LMM-LLR Package of Project Mol-Infer"
date: "July 4, 2021"
author: "Discrete Mathematics Lab, Kyoto University"
---

**STATUS**

We have carefully prepared this repository. If one finds bugs or mistakes, please contact us so that we can fix them
in the next release. Thank you.

---

## General info: mol-infer: Molecular Infering

Mol-infer is a project developed by the Discrete Mathematics Lab at Kyoto Univerisity (ku-dml). See [the top page](https://github.com/ku-dml/mol-infer) for more detail.

## Introduction of the 2LMM-LLR Package

This package consists of four modules.

+ Module 1 calculates descriptors. See [Module 1](Module_1/) for detail. 
+ Module 2 constructs a prediction function by using *Lasso Linear Regression* (LLR). See [Module 2](Module_2/) for detail.
+ Module 3 implements a *Mixed-Integer Linear Programming* (MILP) that solves the inverse ANN problem.
[Module 3](Module_3/) for detail.
+ Module 4 generates graphs (partial enumeration). See [Module 4](Module_4/) for detail.

In order to understand how they deal with these tasks, one may need to read our [paper](https://arxiv.org/abs/2107.02381).

## Quickstart visual guide

Please check the image below on how data and files are used and passed through different modules
of the 2LMM-LLR Package.
For more details on usage and compiling, please see the user's manual in each module

![Data flow illustration](/2LMM-LLR/doc/2LMM-LLR_flow.PNG)


## Quickstart
A quickstart in Linux:
Module 1
First need to assign FV_2LMM_V018 excutable, then we can run the Module 1
```script
$ cd mol-infer-master/2LMM-LLR
$ cd bin/linux
$ ./FV_2LMM_V018
$ ./FV_2LMM_V018 ../../src/Module_1/sample_instance/sample1.sdf  output.csv
```

Module 2
```script
$ cd ../..
$ cd src/Module_2
$ python lasso_eval_linreg.py sample_instance/FV_Alpha_desc_norm.csv   sample_instance/Alpha_norm_values.txt output 0.01
```

Module 3
```script
$ cd ../..
$ cd src/Module_3
$ python infer_2LMM_LLR.py output 5 42  sample_instance/instance_b4_test_2LMM.txt sample_instance/ins_b4_test_fringe_2LMM.txt  test_Hc_b4_test_1900_1920.sdf
```
Module 4 core dumped

Module 4
```script
$ cd ../..
$ cd src/Module_4/main
$ make generate_isomers
$ ./generate_isomers ../sample_instance/sample.sdf  a b c d e f  output.sdf  ../sample_instance/sample_fringe_tree.txt  ../sample_instance/sample_partition.txt
```
