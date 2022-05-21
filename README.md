## Tracker
#### MLP-MHCA-VOT ####

#### Models (./workSpace/checkpoints/ltr/mlp-mhca/mlp-mhca) [**Baidu(votf)**](https://pan.baidu.com/s/1-ovUIAzDC5VpiJflldb9jQ) ####
#### Raw Results(./results) ####


## Installation
This document contains detailed instructions for installing the necessary dependencied for **MLP-MHCA**. The instructions 
have been tested on Ubuntu 18.04 system.

#### Install dependencies
* Create and activate a conda environment 
    ```bash
    conda create -n mlp-mhca python=3.7
    conda activate mlp-mhca
    ```  
* Install PyTorch
    ```bash
    conda install -c pytorch pytorch=1.5 torchvision=0.6.1 cudatoolkit=10.2
    ```  

* Install other packages
    ```bash
    conda install matplotlib pandas tqdm
    pip install opencv-python tb-nightly visdom scikit-image tikzplotlib gdown
    conda install cython scipy
    sudo apt-get install libturbojpeg
    pip install pycocotools jpeg4py
    pip install wget yacs
    pip install shapely==1.6.4.post2
    ```  
* Setup the environment                                                                                                 
Create the default environment setting files.

    ```bash
    # Change directory to <PATH_of_MLP-MHCA>
    cd MLP-MHCA-VOT
    
    # Environment settings for pytracking. Saved at pytracking/evaluation/local.py
    python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
    
    # Environment settings for ltr. Saved at ltr/admin/local.py
    python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
    ```
You can modify these files to set the paths to datasets, results paths etc.
* Add the project path to environment variables  
Open ~/.bashrc, and add the following line to the end. Note to change <path_of_MLP-MHCA> to your real path.
    ```
    export PYTHONPATH=<path_of_MLP-MHCA>:$PYTHONPATH
    ```

## Quick Start
#### Traning
* Modify [local.py](ltr/admin/local.py) to set the paths to datasets, results paths etc.
* Runing the following commands to train the MLP-MHCA. You can customize some parameters by modifying [mlp_mhca.py](ltr/train_settings/mlp-mhca/mlp_mhca.py)
    ```bash
    conda activate mlp-mhca
    cd MLP-MHCA-VOT/ltr
    python run_training.py mlp-mhca mlp_mhca
    ```  

#### Evaluation

* We integrated [PySOT](https://github.com/STVIR/pysot) for evaluation. You can download json files in [PySOT](https://github.com/STVIR/pysot)
    
    You need to specify the path of the model and dataset in the [test.py](pysot_toolkit/test.py).
    ```python
    net_path = '/path_to_model' #Absolute path of the model
    dataset_root= '/path_to_datasets' #Absolute path of the datasets
    ```  
    Then run the following commands.
    ```bash
    conda activate mlp-mhca
    cd MLP-MHCA-VOT
    python -u pysot_toolkit/test.py --dataset <name of dataset> --name 'mlp-mhca' #test tracker 
    python pysot_toolkit/eval.py --tracker_path results/ --dataset <name of dataset> --num 1 --tracker_prefix 'mlp-mhca' #eval tracker
    ```  
    The testing results will in the current directory(./results/dataset/mlp-mhca/)
    

#### Tune

* For the most suitable hyperparameters for the tracker, we provide script to seach automatically
    ```bash
    conda activate mlp-mhca
    cd MLP-MHCA-VOT/pysot_toolkit
    python tune.py --dataset <name of dataset>
    ```  
    The tuned results will in the current directory(./pysot_toolkit/tune_results)


* You can also use [pytracking](pytracking) to test and evaluate tracker. 
The results might be slightly different with [PySOT](https://github.com/STVIR/pysot) due to the slight difference in implementation (pytracking saves results as integers, pysot toolkit saves the results as decimals).


This is a modified version of the python framework [TransT](https://github.com/chenxin-dlut/TransT) based on **Pytorch**, 
also borrowing from [PySOT](https://github.com/STVIR/pysot). 
We would like to thank their authors for providing great frameworks and toolkits.

## Contact
* Sun Shipeng (email:983082671@qq.com)
