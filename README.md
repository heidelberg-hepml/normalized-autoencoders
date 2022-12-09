# A Normalized Autoencoder for LHC triggers

We want to use the Normalised AutoEncoder (NAE) to detect anomalous jets or events measured at the LHC.

The idea was originally put forward in this paper:

https://arxiv.org/pdf/2105.05735.pdf </br>
**Autoencoding under normalization constraints**</br>
*Sangwoong Yoon, Yung-Kyun Noh and Frank C. Park*</br>
https://www.youtube.com/watch?v=ra6usGKnPGk

and applied on jet images by our group on arXiv:

https://arxiv.org/abs/2206.14225 </br>
**A Normalized Autoencoder for LHC Triggers** </br>
*B. Dillon et al.* </br>

The dataset classes are saved in `loaders/` while the architectures in `models/modules.py`.

