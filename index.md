---
title: Theoretical aspects of ML
header-includes:
    <meta name="viewport" content="initial-scale=1.0, minimum-scale=1.0, maximum-scale=1.0, user-scalable=no" />
---


- - - -
## Deep Neural Networks

### Supervised Deep Learning
#### Representation
* Deeper networks can represent functions with exponentially more "kinks" 
  - [https://arxiv.org/pdf/1402.1869.pdf](https://arxiv.org/pdf/1402.1869.pdf)
  - [http://proceedings.mlr.press/v49/telgarsky16.pdf](http://proceedings.mlr.press/v49/telgarsky16.pdf)
* There exist functions that can be represented efficiently with 3 layers, but not with 2 layers 
  - [https://arxiv.org/pdf/1512.03965.pdf](https://arxiv.org/pdf/1512.03965.pdf)
* Deep networks break curse of dimensionality 
  - [https://cbmm.mit.edu/sites/default/files/publications/CBMM-Memo-058-v6.pdf](https://cbmm.mit.edu/sites/default/files/publications/CBMM-Memo-058-v6.pdf)

#### Optimisation 
* NP Hardness of learning even small networks.
  - [https://papers.nips.cc/paper/125-training-a-3-node-neural-network-is-np-complete.pdf](https://papers.nips.cc/paper/125-training-a-3-node-neural-network-is-np-complete.pdf)
* Hardness of learning parity like functions and deep networks
  - [https://core.ac.uk/download/pdf/62919210.pdf](https://core.ac.uk/download/pdf/62919210.pdf)
* Failures of Gradient based methods
  - [https://arxiv.org/pdf/1703.07950.pdf](https://arxiv.org/pdf/1703.07950.pdf)
* Loss surface arguments
  - [http://proceedings.mlr.press/v70/pennington17a/pennington17a.pdf](http://proceedings.mlr.press/v70/pennington17a/pennington17a.pdf)
  - [http://proceedings.mlr.press/v38/choromanska15.pdf](http://proceedings.mlr.press/v38/choromanska15.pdf)
  - [https://arxiv.org/pdf/1511.04210.pdf](https://arxiv.org/pdf/1511.04210.pdf)
  - [https://arxiv.org/abs/1703.09833](https://arxiv.org/abs/1703.09833)
  - [http://papers.nips.cc/paper/6112-deep-learning-without-poor-local-minima.pdf](http://papers.nips.cc/paper/6112-deep-learning-without-poor-local-minima.pdf)
* Reparameterisation/Normalisation approaches:
  - Batch Norm: [https://arxiv.org/pdf/1805.11604.pdf](https://arxiv.org/pdf/1805.11604.pdf)
  - Weight norm: 
  - Layer Norm: 
  - Path Norm : [https://arxiv.org/pdf/1506.02617.pdf](https://arxiv.org/pdf/1506.02617.pdf)
  - Natural Gradient: [https://arxiv.org/pdf/1711.01530.pdf](https://arxiv.org/pdf/1711.01530.pdf)
* Weight sharing (ala CNNs) help in optimisation
  - [https://arxiv.org/abs/1802.02547](https://arxiv.org/abs/1802.02547)
  - [https://arxiv.org/pdf/1709.06129.pdf](https://arxiv.org/pdf/1709.06129.pdf)
* Deep Linear Networks for understanding optimisation:
  - [https://arxiv.org/pdf/1312.6120.pdf](https://arxiv.org/pdf/1312.6120.pdf)
  - [http://proceedings.mlr.press/v80/laurent18a/laurent18a.pdf](http://proceedings.mlr.press/v80/laurent18a/laurent18a.pdf)
* Deep nets learn for linearly-separable and other types of “structured data” or in “PAC setting”
  - [https://arxiv.org/pdf/1710.10174.pdf](https://arxiv.org/pdf/1710.10174.pdf)
  - [https://arxiv.org/abs/1808.01204](https://arxiv.org/abs/1808.01204)
  - [https://arxiv.org/pdf/1712.00779.pdf](https://arxiv.org/pdf/1712.00779.pdf)
* Overparameterised deep nets “converge”
  - [https://arxiv.org/pdf/1811.03962.pdf](https://arxiv.org/pdf/1811.03962.pdf)
* Using ResNets instead of standard deep nets help in optimisation
  - fill-in
  
#### Generalisation
* Size of weights, not number of parameters defines complexity
  - [http://www.yaroslavvb.com/papers/bartlett-sample.pdf](http://www.yaroslavvb.com/papers/bartlett-sample.pdf)
* VC dimension of Neural Nets.
  - [http://mathsci.kaist.ac.kr/~nipl/mas557/VCD_ANN_3.pdf](http://mathsci.kaist.ac.kr/~nipl/mas557/VCD_ANN_3.pdf)
* Spectral Normalised Margin for deep networks.
  - [https://arxiv.org/pdf/1706.08498.pdf](https://arxiv.org/pdf/1706.08498.pdf)
* Compression properties of deep networks.
  - [https://arxiv.org/pdf/1802.05296.pdf](https://arxiv.org/pdf/1802.05296.pdf)
    
#### Other Paradigms for Understanding Deep Networks
* Information Bottleneck
  - [https://arxiv.org/pdf/1703.00810.pdf](https://arxiv.org/pdf/1703.00810.pdf)
* Random weights
  - [https://arxiv.org/pdf/1504.08291.pdf](https://arxiv.org/pdf/1504.08291.pdf)
* Sum-Product networks
  - [https://arxiv.org/pdf/1705.02302.pdf](https://arxiv.org/pdf/1705.02302.pdf)
  - [https://papers.nips.cc/paper/4350-shallow-vs-deep-sum-product-networks.pdf](https://papers.nips.cc/paper/4350-shallow-vs-deep-sum-product-networks.pdf)
* Convolutional nets learning filters/scattering networks/sparse coding for images
  - [https://arxiv.org/pdf/1512.06293.pdf](https://arxiv.org/pdf/1512.06293.pdf)
  - [https://papers.nips.cc/paper/5348-convolutional-kernel-networks.pdf](https://papers.nips.cc/paper/5348-convolutional-kernel-networks.pdf)
  - [https://arxiv.org/pdf/1601.04920.pdf](https://arxiv.org/pdf/1601.04920.pdf)
  - [https://arxiv.org/pdf/1203.1513.pdf](https://arxiv.org/pdf/1203.1513.pdf)
* Improper learning using Kernels
  - [https://arxiv.org/abs/1510.03528](https://arxiv.org/abs/1510.03528)
* SGD/Architecture/Initialisation together to define a “hypothesis class”
  - [https://arxiv.org/pdf/1702.08503.pdf](https://arxiv.org/pdf/1702.08503.pdf)
  - [https://arxiv.org/pdf/1602.05897.pdf](https://arxiv.org/pdf/1602.05897.pdf)
* “Convexified” neural networks
  - [http://jmlr.org/papers/volume18/14-546/14-546.pdf](http://jmlr.org/papers/volume18/14-546/14-546.pdf)
  - [https://arxiv.org/pdf/1609.01000.pdf](https://arxiv.org/pdf/1609.01000.pdf)


- - - -
