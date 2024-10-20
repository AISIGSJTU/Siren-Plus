# SIREN<sup>+</sup>: Robust Federated Learning with Proactive Alarming and Differential Privacy

This code accompanies the paper 'SIREN<sup>+</sup>: Robust Federated Learning with Proactive Alarming and Differential Privacy', which is accepted by *IEEE Transactions on Dependable and Secure Computing (2024)*.

Please download Fashion-MNIST dataset to path like ```/home/data/``` on the user's machine if you want to use it. CIFAR-10 dataset can be downloaded by the program automatically.

Recommended Dependencies: We have tested our system using the following environment configuration.

```
conda create -n siren-plus python=3.8.12
conda activate siren-plus
conda install tensorflow-gpu==2.4.1
pip install tensorflow-privacy==0.5.1
conda install keras==2.3.1 scikit-learn pandas
```

To run the code without any customized settings, please use:

```english
python main.py
```

While if you want to run the code successfully with your customized parameters, you also need to set the following basic hyper-parameters:

| Parameter   | Function                                               |
| ----------- | ------------------------------------------------------ |
| --gar       | Gradient Aggregation Rule                              |
| --eta       | Learning Rate                                          |
| --k         | Number of agents                                       |
| --C         | Fraction of agents chosen per time step                |
| --E         | Number of epochs for each agent                        |
| --T         | Total number of iterations                             |
| --B         | Batch size at each agent                               |
| --mal_obj   | Single or multiple targets                             |
| --mal_num   | Number of targets                                      |
| --mal_strat | Strategy to follow                                     |
| --mal_boost | Boosting factor                                        |
| --mal_E     | Number of epochs for malicious agent                   |
| --ls        | Ratio of benign to malicious steps in alt. min. attack |
| --rho       | Weighting factor for distance constraint               |
| --attack_type| attack type of malicious clients                      |
| --mia       | use membership inference attack                        |
| --malicious_proportion| the proportion of malicious clients in the system|
| --non_iidness| the non-iidness of the data distribution on the clients |

and SIREN<sup>+</sup> exclusive parameters (if you want to use SIREN<sup>+</sup>):

| Parameter         | Function                                               |
| -----------       | ------------------------------------------------------ |
| --server_c        | threshold $C_s$ used by the server                     |
| --client_c        | threshold $C_c$ used by the client                     |
| --server_prohibit | threshold to trigger the penalty mechanism.            |
| --forgive         | the award value used by the award mechanism            |
| --root_size       | the size of the root test dataset                      |
| --dp              | train local models with LDP on all the benign clients  |
| --dp_mul          | noise multiplier for LDP                               |

For more parameters and details of the above parameters, please refer to ```\global_vars.py```

If you want to use the same settings as us, here are some examples:

To run federated training with 10 agents and averaging based aggregation with Fashion-MNIST dataset, use

```english
python main.py --dataset=fMNIST --k=10 --C=1.0 --E=5 --T=40 --B=64 --train --model_num=1 --gar=avg
```
While if you want to use CIFAR-10 dataset and ResNet-18, please set --dataset=CIFAR-10 and --model_num=9.

To run SIREN<sup>+</sup> under label-flipping attack with Fashion-MNIST dataset using LDP, use

```
python main.py --dataset=fMNIST --k=10 --C=1.0 --E=5 --T=40 --B=64 --train --model_num=1 --mal --gar=siren --attack_type=label_flipping --dp --dp_mul=1.0
```

After running the code, please check ```/output``` directory for the results (please manually create the ```output``` directory before the execution of the codes).

To cite our paper, please use the following BibTex:
```
@article{guo2024siren+,
  title={SIREN+: Robust Federated Learning with Proactive Alarming and Differential Privacy},
  author={Guo, Hanxi and Wang, Hao and Song, Tao and Hua, Yang and Ma, Ruhui and Jin, Xiulang and Xue, Zhengui and Guan, Haibing},
  journal={IEEE Transactions on Dependable and Secure Computing (TDSC)},
  year={2024},
  publisher={IEEE}
}
```
