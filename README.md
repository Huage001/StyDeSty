# StyDeSty

This is the pytorch implementation of the following ICML 2024 paper:

**StyDeSty: Min-Max Stylization and Destylization for Single Domain Generalization**

*Songhua Liu, Xin Jin, Xingyi Yang, Jingwen Ye, and Xinchao Wang.*

<img src="https://github.com/Huage001/StyDeSty/assets/fig1.png" width="1024px"/>

<img src="https://github.com/Huage001/StyDeSty/assets/fig2.png" width="1024px"/>

> Single domain generalization (single DG) aims at learning a robust model generalizable to unseen domains from only one training domain, making it a highly ambitious and challenging task. State-of-the-art approaches have mostly relied on data augmentations, such as adversarial perturbation and style enhancement, to synthesize new data and thus increase robustness. Nevertheless, they have largely overlooked the underlying coherence between the augmented domains, which in turn leads to inferior results in real-world scenarios. In this paper, we propose a simple yet effective scheme, termed as **StyDeSty**, to explicitly account for the alignment of the source and pseudo domains in the process of data augmentation, enabling them to interact with each other in a self-consistent manner and further giving rise to a latent domain with strong generalization power. The heart of StyDeSty lies in the interaction between a **stylization** module for generating novel stylized samples using the source domain, and a **destylization** module for transferring stylized and source samples to a latent domain to learn content-invariant features. The stylization and destylization modules work adversarially and reinforce each other. During inference, the destylization module transforms the input sample with an arbitrary style shift to the latent domain, in which the downstream tasks are carried out. Specifically, the location of the destylization layer within the backbone network is determined by a dedicated neural architecture search (NAS) strategy. We evaluate StyDeSty on multiple benchmarks and demonstrate that it yields encouraging results, outperforming the state of the art by up to 13.44% on classification accuracy.

**Note: This work was conducted in 2022 spring.**

## Installation

* Create a new environment if you want:

  ```bash
  conda create -n StyDeSty python=3.9
  conda activate StyDeSty
  ```

* Clone the repo:

    ```bash
    git clone https://github.com/Huage001/StyDeSty.git
    cd StyDeSty
    ```

* Install *torch* and *torchvision* following the [official instruction](https://pytorch.org/get-started/locally/). For example, in Linux, you can use the following command:

   ```bash
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
   ```

## Single Domain Generalization

* Make a directory containing datasets:

  ```bash
  mkdir data
  ```

### PACS Dataset

* Download *PACS.zip* from [Google Drive](https://drive.google.com/file/d/1j92QlahBv9DpGqSR0RJskUofVwhlkJM6/view?usp=share_link), move it into *data* directory, and unzip it:

  ```bash
  unzip data/PACS.zip -d data
  rm data/PACS.zip
  ```

* Run *script/run_P2ACS.sh* to train on the Photo domain and test on the Art Painting, Cartoon, and Sketch domains:

  ```bash
  CUDA_VISIBLE_DEVICES=0 bash script/run_P2ACS.sh
  # run script/run_A2PCS.sh, script/run_C2PAS.sh, script/run_S2PAC.sh to use other domains for training
  ```

* *network.py* already contains the optimal location for the AdaIN layer. If you want to run the NAS algorithm to search for the optimal location, run *script/run_nas_P2ACS.sh*:

  ```bash
  CUDA_VISIBLE_DEVICES=0 bash script/run_nas_P2ACS.sh
  # run script/run_nas_A2PCS.sh, script/run_nas_C2PAS.sh, script/run_nas_S2PAC.sh to use other domains for training
  ```

### Digits Dataset

* Download *Digits.zip* from [Google Drive](https://drive.google.com/file/d/1iLCSmCLRpIbng5X2ZxykjBbRMjNbS5X4/view?usp=share_link), move it into *data* directory, and unzip it:

  ```bash
  unzip data/Digits.zip -d data
  rm data/Digits.zip
  ```

* Run *script/run_Digits.sh* to train on MNIST and test on SVHN, M-MNIST, SYN, and USPS:

  ```bash
  CUDA_VISIBLE_DEVICES=0 bash script/run_Digits.sh
  ```

* *network.py* already contains the optimal location for the AdaIN layer. If you want to run the NAS algorithm to search for the optimal location, run *script/run_nas_Digits.sh*:

  ```bash
  CUDA_VISIBLE_DEVICES=0 bash script/run_nas_Digits.sh
  ```

### CIFAR10-C Dataset

* Download *CIFAR-10.zip* from [Google Drive](https://drive.google.com/file/d/1TOXvcFTYF2VM1FMJDpf-aLDW4_rcr50p/view?usp=share_link), move it into *data* directory, and unzip it:

  ```bash
  unzip data/CIFAR-10.zip -d data
  rm data/CIFAR-10.zip
  ```

* Run *script/run_CIFAR.sh* to train on the clean CIFAR and test on CIFAR10-C:

  ```bash
  CUDA_VISIBLE_DEVICES=0 bash script/run_CIFAR.sh
  ```

* *network.py* already contains the optimal location for the AdaIN layer. If you want to run the NAS algorithm to search for the optimal location, run *script/run_nas_CIFAR.sh*:

  ```bash
  CUDA_VISIBLE_DEVICES=0 bash script/run_nas_CIFAR.sh
  ```

## Acknowledgement

Some codes and hyper-parameters are borrowed from [Learning_to_diversify](https://github.com/BUserName/Learning_to_diversify).

## Citation

If you find this project useful in your research, please consider citing:

```latex
@article{liu2024stydesty,
    author    = {Songhua Liu, Xin Jin, Xingyi Yang, Jingwen Ye, Xinchao Wang},
    title     = {StyDeSty: Min-Max Stylization and Destylization for Single Domain Generalization},
    journal   = {International Conference on Machine Learning},
    year      = {2024},
}
```