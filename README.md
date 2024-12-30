# DSPProject

This repository contains the code for the HeSBO, PCA-BO and DSP algorithms.

## Table of content

- [Project description](#project-description)
- [Arguments](#arguments)
    * [Example command](#example-command)
- [File specification](#file-specifications)
- [Contact](#contact)

## Project description


## Installation
To install the program, simply pull this repo and use the following 
command to run the program. Of course, you will need to add your own arguments.

``` 
git pull https://github.com/Skippybal/DSPProject.git
```

After pulling you can run the algorithm using the following files:

| algorithm         | File                      |
| ---               |---------------------------| 
| DSP               | main_2.py                 |                     
| PCA-BO                | pca-bo-org.py             | 
| DSP-PCABO       | dsp-pca-bo_lbfgs.py             |    
| HeSBO          | HeSBO/count_sketch.py     | 
| DSP-HeSBO    | HeSBO/count_sketch_dsp.py |




## Arguments
| Parameter | Description                                  | Type   | Default |
| ---               |----------------------------------------------|--------|---------|
| --folder_name | Folder to store results in                   | String | ``run`` |
| --algo_name | Algorithm name                               | String |         |
| --dims | Dimensionality of the problem                | Int    |         |
| --func | BBoB function ID                             | Int |         |
| --n_trails | Number of repetitions                        | Int | 5       |
| --doe | Number of DoE points                         | Int | 20      |
| --total | Total number of function evalutaion (budget) | Int |         |


### Example command
This is and example of a command that can be used to test the project. 
```
python .\pca-bo-org.py --folder_name PCA-BO --func 8 --total 1000 --doe 100 --dims 100 --algo_name PCA-BO
```


## Contact

* K. Notebomer
* k.a.notebomer@vuw.lei