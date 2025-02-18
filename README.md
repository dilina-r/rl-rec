# 1. Introduction

This repository provides the complete code for data preprocessing, training and testing several recommenders, based on our paper "Reassessing the Effectiveness of Reinforcement Learning based Recommender Systems for Sequential Recommendation" by Dilina Rajapakse and Dietmar Jannach (Submitted to SIGIR 2025).

For full results, hyperparameters and additional information, visit our [Online Material](https://dilina-r.github.io/rl-rec/) page.

The algorithms and pre-processing methods are based on existing code developed and shared by:

- SKNN and VSKNN implementation from the session-rec framework [Original Code](https://github.com/rn5l/session-rec)
- GRU4Rec [1,2] official PyTorch implementation from the Authors [Original Code](https://github.com/hidasib/GRU4Rec_PyTorch_Official)
- NARM [3] PyTorch implementation by Wang-Shuo [Original Code](https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch/)
- SQN [4] implementation from the Authors [Original Code](https://drive.google.com/file/d/1nLL3_knhj_RbaP_IepBLkwaT6zNIeD5z/view)
- EVAL [5] implementation from the Authors [Original Code](https://github.com/alfa-labarca/RL-Proxy-Models)


# 2. Installation

## Conda environment setup

Below environments require Anaconda to be installed. Download and Install from here (https://www.anaconda.com/distribution/)

### (i) KNN approaches

- We used the session-rec framework. Please refer to [session-rec](https://github.com/rn5l/session-rec) 

### (ii) GRU4Rec and NARM approaches (PyTorch)

- Run the following command to create the GPU conda environment.

    `conda create --name pytorch python=3.11`

- Once the environment is created, install the required python packages.

    ```
    conda activate pytorch
    pip install -r requirements_gpu.txt
    ```

### (iii) RL baselines and SQN, EVAL approaches

- Since the SQN implementations are based on older Tensorflow 1.x versions, these cannot be run on newer GPUs. To create the CPU conda environment, run:

    `conda create --name tf1 python=3.6`

- Once the environment is created, install the required Python packages.

    ```
    conda activate tf1
    conda install -c conda-forge cmake dm-tree
    pip install -r requirements_tf1.txt
    ```

- If an error occurs when installing the *trfl* and *dm-tree* packages, install *dm-tree* using the command below, and proceed to install the remaining packages separately.

    `pip install dm-tree --only-binary :all:`


# 3. Data Pre-processing

- The original raw datasets can be downloaded from this [Google Drive](https://drive.google.com/drive/folders/1ritDnO_Zc6DFEU6UND9C8VCisT0ETVp5)

- Pre-processing scripts for the datasets Yoochoose, RetailRocket, Diginetica are provided both for the *GRU4Rec-protocol* and *SQN-protocol* in the `preprocess` directory. For example, to pre-process the datasets and generate the processed data splits for the Yoochoose dataset with *GRU4Rec-protocol*:

    `python preprocess/gru_protocol/preprocess_yoochoose.py --src <path to raw datasaet dir> --dst <path to output directory>`

- To generate the replay_buffer for training the RL-methods, run the script and provide the directory path of the processed datasets

    `python preprocess/create_replay_buffer.py --path <path to output directory>`

- The `run_preprocess.sh` script includes all the commands to generate the data-splits for the three datasets (Yoochoose, RetailRocket, Diginetica) using both the *GRU4Rec-protocol* and *SQN-protocol* protocols. Before running the script, provide the directory paths of the raw datasets into the respective parameters as instructed in the file comments. 

# 4. Running the Code

- To run each algorithm, please run the respective scipts provided, in the respective conda environments. 

    ```
    bash run_gru4rec.sh
    bash run_narm.sh
    bash run_rl.sh
    ```

- The **run_rl.sh** script contains the commands for each RL baseline and the RL-enhanced SQN and EVAL methods. Comment/uncomment as needed to run each approach.

- To run the SKNN and VSKNN methods, we used the [session-rec](https://github.com/rn5l/session-rec) framework with the processed datasets. To reproduce our results, refer to our [Online material](https://dilina-r.github.io/rl-rec/), where we provide the full results and hyperparameters for each protocol.

# References

[1] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, Domonkos Tikk: Session-based Recommendations with Recurrent Neural Networks, ICLR 2016

[2] Balázs Hidasi, Alexandros Karatzoglou: Recurrent Neural Networks with Top-k Gains for Session-based Recommendations, CIKM 2018

[3] Jing Li, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Jun Ma: Neural Attentive Session-based Recommendation, CIKM 2017

[4] Xin Xin, Alexandros Karatzoglou, Ioannis Arapakis, Joemon M. Jose: Self-Supervised Reinforcement Learning for Recommender Systems, SIGIR 2020

[5] Álvaro Labarca Silva, Denis Parra, Rodrigo Toro Icarte: On the Unexpected Effectiveness of Reinforcement Learning for Sequential Recommendation, ICML 2024

