# awsome-domain-adaptation

This repo is a collection of AWESOME things about domian adaptation,including papers,code etc.Feel free to star and fork.

# Contents
- [Papers](#papers)
  - [Overview](#overview)
  - [Theory](#theory)
  - [Unsupervised DA](#unsupervised-da)
    - [Adversarial Methods](#adversarial-methods)
    - [Network Methods](#network-methods)
    - [Optimal Transport](#optimal-transport)
    - [Incremental Methods](#incremental-mehtods)
    - [Other Methods](#other-methods)
  - [Weakly-Supervised DA](#weakly-supervised-da)
  - [Zero-shot DA](#zero-shot-da)
  - [One-shot DA](#one-shot-da)
  - [Few-shot DA](#few-shot-da)
  - [Image-to-Image Translation](#image-to-image-translation)
  - [Disentangled Representation Learning](#disentangled-representation-learning)
  - [Open Set DA](#open-set-da)
  - [Partial DA](#partial-da)
  - [Multi Source DA](#multi-source-da)
  - [Muti Target DA](#multi-target-da)
  - [Multi Step DA](#multi-step-da)
  - [General Transfer Learning](#general-transfer-learning)
    - [Domain Generalization](#domain-generalization)
    - [Meta-Learning](#meta-learning)
    - [Transfer Metric Learning](#transfer-metric-learning)
    - [Others](#others)
  - [Applications](#applications)
    - [Object Detection](#object-detection)
    - [Semantic Segmentation](#semantic-segmentation)
    - [Person Re-Identification](#person-re-identification)
    - [Medical Related](#medical-related)
    - [Others](#others)
  - [Benchmarks](#benchmarks)
- [Code](#code)
- [Other Resources](#other-resources)

# Papers
## Overview
- An introduction to domain adaptation and transfer learning [[arXiv 31 Dec 2018]](https://arxiv.org/abs/1812.11806v2)
- Adversarial Transfer Learning [[arXiv 6 Dec 2018]](https://arxiv.org/abs/1812.02849v1)
- A Survey on Deep Transfer Learning [[ICANN2018]](https://arxiv.org/abs/1808.01974v1)
- Deep Visual Domain Adaptation: A Survey [[arXiv 2018]](https://arxiv.org/abs/1802.03601v4)
- Transfer Learning for Cross-Dataset Recognition: A Survey [[arXiv 2017]](https://sci-hub.tw/https://arxiv.org/abs/1705.04396)
- Domain Adaptation for Visual Applications: A Comprehensive Survey  [[arXiv 2017]](https://arxiv.org/abs/1702.05374)
- Visual domain adaptation: A survey of recent advances [[2015]](https://sci-hub.tw/10.1109/msp.2014.2347059)

## Theory
- On Learning Invariant Representation for Domain Adaptation [[arXiv on 27 Jan 2019]](https://arxiv.org/abs/1901.09453v1)
- Theoretical Perspective of Deep Domain Adaptation [[arXiv 15 Nov 2018]](https://arxiv.org/abs/1811.06199v2)
- A theory of learning from different domains [[ML2010]](https://link.springer.com/content/pdf/10.1007%2Fs10994-009-5152-4.pdf)
- Learning Bounds for Domain Adaptation [[NIPS2007]](http://papers.nips.cc/paper/3212-learning-bounds-for-domain-adaptation)
- Analysis of Representations for Domain Adaptation [[NIPS2006]](https://papers.nips.cc/paper/2983-analysis-of-representations-for-domain-adaptation)
 
## Unsupervised DA
### Adversarial Methods
- Cluster Alignment with a Teacher for Unsupervised Domain Adaptation [[arXiv 24 Mar 2019]](https://arxiv.org/abs/1903.09980v1)
- Domain-Invariant Adversarial Learning for Unsupervised Domain Adaption [[arXiv 30 Nov 2018]](https://arxiv.org/abs/1811.12751)
- Consensus Adversarial Domain Adaptation [[AAAI2019]](https://aaai.org/Papers/AAAI/2019/AAAI-ZouH.697.pdf)
- Unsupervised Domain Adaptation using Deep Networks with Cross-Grafted Stacks [[arXiv 17 Feb 2019]](https://arxiv.org/abs/1902.06328v1)
- DART: Domain-Adversarial Residual-Transfer Networks for Unsupervised Cross-Domain Image Classification [[arXiv 30 Dec 2018]](https://arxiv.org/abs/1812.11478)
- Progressive Feature Alignment for Unsupervised Domain Adaptation [[arXiv 21 Nov 2018]](https://arxiv.org/abs/1811.08585v1)
- Augmented Cyclic Adversarial Learning for Low Resource Domain Adaptation [[ICLR2019]](https://openreview.net/forum?id=B1G9doA9F7)
- Transferable Attention for Domain Adaptation [[AAAI2019]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transferable-attention-aaai19.pdf)
- Conditional Adversarial Domain Adaptation [[NIPS2018]](http://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation) [[Pytorch(official)]](https://github.com/thuml/CDAN)  [[Pytorch(third party)]](https://github.com/thuml/CDAN)
- Unsupervised Domain Adaptation using Generative Models and Self-ensembling [[arXiv 2 Dec 2018]](https://arxiv.org/abs/1812.00479)
- Exploiting Local Feature Patterns for Unsupervised Domain Adaptation [[AAAI2019]](https://arxiv.org/abs/1811.05042v2)
- Domain Confusion with Self Ensembling for Unsupervised Adaptation [[arXiv 10 Oct 2018]](https://arxiv.org/abs/1810.04472)
- Improving Adversarial Discriminative Domain Adaptation [[arXiv 10 Sep 2018]](https://arxiv.org/abs/1809.03625)
- M-ADDA: Unsupervised Domain Adaptation with Deep Metric Learning [[arXiv 6 Jul 2018]](https://arxiv.org/abs/1807.02552v1) [[Pytorch(official)]](https://github.com/IssamLaradji/M-ADDA)
- Factorized Adversarial Networks for Unsupervised Domain Adaptation [[arXiv 4 Jun 2018]](https://arxiv.org/abs/1806.01376v1)
- DiDA: Disentangled Synthesis for Domain Adaptation [[arXiv 21 May 2018]](https://arxiv.org/abs/1805.08019v1)
- Unsupervised Domain Adaptation with Adversarial Residual Transform Networks [[arXiv 25 Apr 2018]](https://arxiv.org/abs/1804.09578)
- Causal Generative Domain Adaptation Networks [[arXiv 28 Jun 2018]](https://arxiv.org/abs/1804.04333v3)
- Semi-supervised Adversarial Learning to Generate Photorealistic Face Images of New Identities from 3D Morphable Model [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Baris_Gecer_Semi-supervised_Adversarial_Learning_ECCV_2018_paper.pdf)
- Deep Adversarial Attention Alignment for Unsupervised Domain Adaptation: the Benefit of Target Expectation Maximization [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Guoliang_Kang_Deep_Adversarial_Attention_ECCV_2018_paper.pdf)
- Learning Semantic Representations for Unsupervised Domain Adaptation [[ICML2018]](http://proceedings.mlr.press/v80/xie18c.html) [[TensorFlow(Official)]](https://github.com/Mid-Push/Moving-Semantic-Transfer-Network)
- CyCADA: Cycle-Consistent Adversarial Domain Adaptation [[ICML2018]](http://proceedings.mlr.press/v80/hoffman18a.html) [[Pytorch(official)]](https://github.com/jhoffman/cycada_release)
- From source to target and back: Symmetric Bi-Directional Adaptive GAN [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Russo_From_Source_to_CVPR_2018_paper.pdf) [[Keras(Official)]](https://github.com/engharat/SBADAGAN) [[Pytorch]](https://github.com/naoto0804/pytorch-SBADA-GAN)
- Detach and Adapt: Learning Cross-Domain Disentangled Deep Representation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Detach_and_Adapt_CVPR_2018_paper.pdf) [[Tensorflow]](https://github.com/ycliu93/CDRD)
- Maximum Classifier Discrepancy for Unsupervised Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf) [[Pytorch(Official)]](https://github.com/mil-tokyo/MCD_DA)
- Adversarial Feature Augmentation for Unsupervised Domain Adaptation [[CVPR2018]](https://arxiv.org/abs/1711.08561) [[TensorFlow(Official)]](https://github.com/ricvolpi/adversarial-feature-augmentation)
- Duplex Generative Adversarial Network for Unsupervised Domain Adaptation [[CVPR2018]](http://vipl.ict.ac.cn/uploadfile/upload/2018041610083083.pdf) [[Pytorch(Official)]](http://vipl.ict.ac.cn/view_database.php?id=6)
- Generate To Adapt: Aligning Domains using Generative Adversarial Networks [[CVPR2018]](https://arxiv.org/abs/1704.01705) [[Pytorch(Official)]](https://github.com/yogeshbalaji/Generate_To_Adapt)
- Image to Image Translation for Domain Adaptation [[CVPR2018]](https://arxiv.org/abs/1712.00479)
- Unsupervised Domain Adaptation with Similarity Learning [[CVPR2018]](https://arxiv.org/abs/1711.08995)
- Conditional Generative Adversarial Network for Structured Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hong_Conditional_Generative_Adversarial_CVPR_2018_paper.pdf) 
- Collaborative and Adversarial Network for Unsupervised Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Collaborative_and_Adversarial_CVPR_2018_paper.pdf) [[Pytorch]](https://github.com/zhangweichen2006/iCAN)
- Re-Weighted Adversarial Adaptation Network for Unsupervised Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Re-Weighted_Adversarial_Adaptation_CVPR_2018_paper.pdf)
- Multi-Adversarial Domain Adaptation [[AAAI2018]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/multi-adversarial-domain-adaptation-aaai18.pdf) [[Caffe(Official)]](https://github.com/thuml/MADA)
- Wasserstein Distance Guided Representation Learning for Domain Adaptation [[AAAI2018]](https://arxiv.org/abs/1707.01217) [[TensorFlow(official)]](https://github.com/RockySJ/WDGRL)
- Incremental Adversarial Domain Adaptation for Continually Changing Environments [[ICRA2018]](https://arxiv.org/abs/1712.07436)
- Adversarial Dropout Regularization [[ICLR2018]](https://openreview.net/forum?id=HJIoJWZCZ)
- A DIRT-T Approach to Unsupervised Domain Adaptation [[ICLR2018 Poster]](https://openreview.net/forum?id=H1q-TM-AW) [[Tensorflow(Official)]](https://github.com/RuiShu/dirt-t)
- Label Efficient Learning of Transferable Representations acrosss Domains and Tasks [[NIPS2017]](http://vision.stanford.edu/pdf/luo2017nips.pdf) [[Project]](http://alan.vision/nips17_website/)
- Adversarial Discriminative Domain Adaptation [[CVPR2017]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf)  [[Tensorflow(Official)]](https://github.com/erictzeng/adda) [[Pytorch]](https://github.com/corenel/pytorch-adda)
- Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks [[CVPR2017]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bousmalis_Unsupervised_Pixel-Level_Domain_CVPR_2017_paper.pdf) [[Tensorflow(Official)]](https://github.com/tensorflow/models/tree/master/research/domain_adaptation) [[Pytorch]](https://github.com/vaibhavnaagar/pixelDA_GAN)
- Domain Separation Networks [[NIPS2016]](http://papers.nips.cc/paper/6254-domain-separation-networks)
- Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation [[ECCV2016]](https://arxiv.org/abs/1607.03516)
- Domain-Adversarial Training of Neural Networks [[JMLR2016]](http://www.jmlr.org/papers/volume17/15-239/15-239.pdf)
- Unsupervised Domain Adaptation by Backpropagation [[ICML2015]](http://proceedings.mlr.press/v37/ganin15.pdf) [[Caffe(Official)]](https://github.com/ddtm/caffe/tree/grl) [[Tensorflow]](https://github.com/shucunt/domain_adaptation) [[Pytorch]](https://github.com/fungtion/DANN)

### Network Methods
- Joint Domain Alignment and Discriminative Feature Learning for Unsupervised Deep Domain Adaptation [[AAAI2019]](https://arxiv.org/abs/1808.09347v2)
- Boosting Domain Adaptation by Discovering Latent Domains [[CVPR2018]](https://arxiv.org/abs/1805.01386)
- Residual Parameter Transfer for Deep Domain Adaptation [[CVPR2018]](https://arxiv.org/abs/1711.07714)
- Deep Asymmetric Transfer Network for Unbalanced Domain Adaptation [[AAAI2018]](http://media.cs.tsinghua.edu.cn/~multimedia/cuipeng/papers/DATN.pdf)
- Deep CORAL: Correlation Alignment for Deep Domain Adaptation [[ECCV2016]](https://arxiv.org/abs/1607.01719)
- Deep Domain Confusion: Maximizing for Domain Invariance [[Arxiv 2014]](https://arxiv.org/abs/1412.3474)

### Optimal Transport
- DeepJDOT: Deep Joint distribution optimal transport for unsupervised domain adaptation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bharath_Bhushan_Damodaran_DeepJDOT_Deep_Joint_ECCV_2018_paper.pdf)
- Joint Distribution Optimal Transportation for Domain Adaptation [[NIPS2017]](http://papers.nips.cc/paper/6963-joint-distribution-optimal-transportation-for-domain-adaptation.pdf) [[python]](https://github.com/rflamary/JDOT) [[Python Optimal Transport Library]](https://github.com/rflamary/POT)

### Incremental Methods
- Incremental Adversarial Domain Adaptation for Continually Changing Environments [[ICRA2018]](https://arxiv.org/abs/1712.07436)
- Continuous Manifold based Adaptation for Evolving Visual Domains [[CVPR2014]](https://people.eecs.berkeley.edu/~jhoffman/papers/Hoffman_CVPR2014.pdf)

### Other Methods
- Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation [[CVPR2019]](https://arxiv.org/abs/1903.04064)
- Unsupervised Domain Adaptation using Feature-Whitening and Consensus Loss [[arXiv 7 Mar 2019]](https://arxiv.org/abs/1903.03215v1)
- Domain Discrepancy Measure Using Complex Models in Unsupervised Domain Adaptation [[arXiv 30 Jan 2019]](https://arxiv.org/abs/1901.10654v1)
- Domain Alignment with Triplets [[arXiv 22 Jan 2019]](https://arxiv.org/abs/1812.00893v2)
- Contrastive Adaptation Network for Unsupervised Domain Adaptation [[arXiv 4 Jan 2019]](https://arxiv.org/abs/1901.00976v1)
- Unsupervised Domain Adaptation: An Adaptive Feature Norm Approach [[arXiv 19 Nov 2018]](https://arxiv.org/abs/1811.07456v1) [[Pytorch(official)]](https://github.com/jihanyang/AFN)
- Deep Discriminative Learning for Unsupervised Domain Adaptation [[arXiv 17 Nov 2018]](https://arxiv.org/abs/1811.07134v1)
- Unsupervised Domain Adaptation for Distance Metric Learning [[ICLR2019]](https://openreview.net/forum?id=BklhAj09K7)
- Co-regularized Alignment for Unsupervised Domain Adaptation [[NIPS2018]](http://papers.nips.cc/paper/8146-co-regularized-alignment-for-unsupervised-domain-adaptation)
- Domain Invariant and Class Discriminative Feature Learning for Visual Domain Adaptation [[TIP 2018]](https://ieeexplore.ieee.org/document/8362753/)
- Unsupervised Domain Adaptation by Mapped Correlation Alignment [[IEEE ACCESS]](https://ieeexplore.ieee.org/abstract/document/8434290/)
- Graph Adaptive Knowledge Transfer for Unsupervised Domain Adaptation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhengming_Ding_Graph_Adaptive_Knowledge_ECCV_2018_paper.pdf)
- Unsupervised Domain Adaptation with Distribution Matching Machines [[AAAI2018]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/distribution-matching-machines-aaai18.pdf)
- Learning to cluster in order to transfer across domains and tasks [[ICLR2018]](https://openreview.net/forum?id=ByRWCqvT-) [[Bolg]](https://mlatgt.blog/2018/04/29/learning-to-cluster/) [[Pytorch]](https://github.com/GT-RIPL/L2C)
- Self-Ensembling for Visual Domain Adaptation [[ICLR2018 Poster]](https://openreview.net/forum?id=rkpoTaxA-)
- Minimal-Entropy Correlation Alignment for Unsupervised Deep Domain Adaptation [[ICLR2018 Poster]](https://openreview.net/forum?id=rJWechg0Z)
- Aligning Infinite-Dimensional Covariance Matrices in Reproducing Kernel Hilbert Spaces for Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Aligning_Infinite-Dimensional_Covariance_CVPR_2018_paper.pdf)
- Associative Domain Adaptation [[ICCV2017]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Haeusser_Associative_Domain_Adaptation_ICCV_2017_paper.pdf) [[TensorFlow]](https://github.com/haeusser/learning_by_association)
- Asymmetric Tri-training for Unsupervised Domain Adaptation [[ICML2017]](http://proceedings.mlr.press/v70/saito17a.html)
- Learning Transferrable Representations for Unsupervised Domain Adaptation [[NIPS2016]](http://papers.nips.cc/paper/6360-learning-transferrable-representations-for-unsupervised-domain-adaptation)

## Weakly-Supervised DA
- Transferable Curriculum for Weakly-Supervised Domain Adaptation [[AAAI2019]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transferable-curriculum-aaai19.pdf)

## Zero-shot DA
- Zero-shot Domain Adaptation Based on Attribute Information [[arXiv 13 Mar 2019]](https://arxiv.org/abs/1903.05312v1)
- Generalized Zero-Shot Learning with Deep Calibration Network [NIPS2018](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-calibration-network-nips18.pdf)
- Zero-Shot Deep Domain Adaptation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kuan-Chuan_Peng_Zero-Shot_Deep_Domain_ECCV_2018_paper.pdf)

## One-shot DA


## Few-shot DA
- Few-Shot Adversarial Domain Adaptation [[NIPS2017]](http://papers.nips.cc/paper/7244-few-shot-adversarial-domain-adaptation)

## Image-to-Image Translation
- MISO: Mutual Information Loss with Stochastic Style Representations for Multimodal Image-to-Image Translation [[arXiv 11 Feb 2019]](https://arxiv.org/abs/1902.03938)
- TraVeLGAN: Image-to-image Translation by Transformation Vector Learning [[arXiv 25 Feb 2019]](https://arxiv.org/abs/1902.09631)
- Unsupervised Attention-guided Image-to-Image Translation [[NIPS2018]](https://papers.nips.cc/paper/7627-unsupervised-attention-guided-image-to-image-translation)
- Image-to-image translation for cross-domain disentanglement [[NIPS2018]](https://papers.nips.cc/paper/7404-image-to-image-translation-for-cross-domain-disentanglement)
- One-Shot Unsupervised Cross Domain Translation [[NIPS2018]](http://papers.nips.cc/paper/7480-one-shot-unsupervised-cross-domain-translation)
- A Unified Feature Disentangler for Multi-Domain Image Translation and Manipulation [[NIPS2018]](http://papers.nips.cc/paper/7525-a-unified-feature-disentangler-for-multi-domain-image-translation-and-manipulation)
- Unsupervised Image-to-Image Translation Using Domain-Specific Variational Information Bound [[NIPS2018]](http://papers.nips.cc/paper/8236-unsupervised-image-to-image-translation-using-domain-specific-variational-information-bound)
- Multi-view Adversarially Learned Inference for Cross-domain Joint Distribution Matching [[KDD2018]](http://www.kdd.org/kdd2018/accepted-papers/view/multi-view-adversarially-learned-inference-for-cross-domain-joint-distribut)
- Improving Shape Deformation in Unsupervised Image-to-Image Translation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Aaron_Gokaslan_Improving_Shape_Deformation_ECCV_2018_paper.pdf)
- NAM: Non-Adversarial Unsupervised Domain Mapping [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yedid_Hoshen_Separable_Cross-Domain_Translation_ECCV_2018_paper.pdf)
- AugGAN: Cross Domain Adaptation with GAN-based Data Augmentation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sheng-Wei_Huang_AugGAN_Cross_Domain_ECCV_2018_paper.pdf)
- Recycle-GAN: Unsupervised Video Retargeting [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Aayush_Bansal_Recycle-GAN_Unsupervised_Video_ECCV_2018_paper.pdf) [[Project]](http://www.cs.cmu.edu/~aayushb/Recycle-GAN/)
- Unsupervised Image-to-Image Translation with Stacked Cycle-Consistent Adversarial Networks [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Minjun_Li_Unsupervised_Image-to-Image_Translation_ECCV_2018_paper.pdf)
- Diverse Image-to-Image Translation via Disentangled Representations [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Hsin-Ying_Lee_Diverse_Image-to-Image_Translation_ECCV_2018_paper.pdf) [[Pytorch(Official)]](https://github.com/HsinYingLee/DRIT/) [[Tensorflow]](https://github.com/taki0112/DRIT-Tensorflow)
- Discriminative Region Proposal Adversarial Networks for High-Quality Image-to-Image Translation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chao_Wang_Discriminative_Region_Proposal_ECCV_2018_paper.pdf)
- Multimodal Unsupervised Image-to-Image Translation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xun_Huang_Multimodal_Unsupervised_Image-to-image_ECCV_2018_paper.pdf) [[Pytorch(Official)]](https://github.com/nvlabs/MUNIT)
- JointGAN: Multi-Domain Joint Distribution Learning with Generative Adversarial Nets [[ICML2018]](http://proceedings.mlr.press/v80/pu18a.html) [[TensorFlow(Official)]](https://github.com/sdai654416/Joint-GAN)
- DA-GAN: Instance-level Image Translation by Deep Attention Generative Adversarial Networks [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ma_DA-GAN_Instance-Level_Image_CVPR_2018_paper.pdf)
- StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation [[CVPR2018]](https://arxiv.org/abs/1711.09020) [[Pytorch(Official)]](https://github.com/yunjey/StarGAN)
- Conditional Image-to-Image Translation [[CVPR2018]](https://arxiv.org/abs/1805.00251v1)
- Toward Multimodal Image-to-Image Translation [[NIPS2017]](https://arxiv.org/abs/1711.11586) [[Project]](https://junyanz.github.io/BicycleGAN/) [[Pyotorch(Official)]](https://github.com/junyanz/BicycleGAN)
- Unsupervised Image-to-Image Translation Networks [[NIPS2017]](http://papers.nips.cc/paper/6672-unsupervised-image-to-image-translation-networks) [[Pytorch(Official)]](https://github.com/mingyuliutw/unit)
- Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks [[ICCV2017(extended version)]](https://arxiv.org/abs/1703.10593v4) [[Pytorch(Official)]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- Image-to-Image Translation with Conditional Adversarial Nets [[CVPR2017]](https://arxiv.org/abs/1611.07004)  [[Project]](https://phillipi.github.io/pix2pix/) [[Pytorch(Official)]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 
- Learning to Discover Cross-Domain Relations with Generative Adversarial Networks [[ICML2017]](https://arxiv.org/abs/1703.05192) [[Pytorch(Official)]](https://github.com/SKTBrain/DiscoGAN)
- Unsupervised Cross-Domain Image Generation [[ICLR2017 Poster]](https://openreview.net/forum?id=Sk2Im59ex) [[TensorFlow]](https://github.com/yunjey/domain-transfer-network)
- Coupled Generative Adversarial Networks [[NIPS2016]](http://papers.nips.cc/paper/6544-coupled-generative-adversarial-networks) [[Pytorch(Official)]](https://github.com/mingyuliutw/cogan)

## Disentangled Representation Learning
- Towards a Definition of Disentangled Representations [[arXiv 5 Dec 2018]](https://arxiv.org/abs/1812.02230)
- Life-Long Disentangled Representation Learning with Cross-Domain Latent Homologies [[NIPS2018]](https://papers.nips.cc/paper/8193-life-long-disentangled-representation-learning-with-cross-domain-latent-homologies)
- Image-to-image translation for cross-domain disentanglement [[NIPS2018]](https://papers.nips.cc/paper/7404-image-to-image-translation-for-cross-domain-disentanglement)

## Open Set DA
- Learning Factorized Representations for Open-set Domain Adaptation [[ICLR2019]](https://openreview.net/pdf?id=SJe3HiC5KX)
- Open Set Domain Adaptation by Backpropagation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kuniaki_Saito_Adversarial_Open_Set_ECCV_2018_paper.pdf) [[Tensorflow]](https://github.com/Mid-Push/Open_set_domain_adaptation) [[Pytorch]](https://github.com/YU1ut/openset-DA)
- Open Set Domain Adaptation [[ICCV2017]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Busto_Open_Set_Domain_ICCV_2017_paper.pdf)

## Partial DA
- TWINs: Two Weighted Inconsistency-reduced Networks for Partial Domain Adaptation [[arXiv 18 Dec 2018]](https://arxiv.org/abs/1812.07405)
- Partial Adversarial Domain Adaptation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhangjie_Cao_Partial_Adversarial_Domain_ECCV_2018_paper.pdf) [[Pytorch(Official)]](https://github.com/thuml/PADA)
- Importance Weighted Adversarial Nets for Partial Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Importance_Weighted_Adversarial_CVPR_2018_paper.html)
- Partial Transfer Learning with Selective Adversarial Networks [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cao_Partial_Transfer_Learning_CVPR_2018_paper.pdf)[[paper weekly]](http://www.paperweekly.site/papers/1388) [[Pytorch(Official) & Caffe(official)]](https://github.com/thuml/SAN)

## Multi Source DA
- Multi-Source Domain Adaptation with Mixture of Experts [[EMNLP2018]](https://arxiv.org/abs/1809.02256) [[Tensorflow]](https://github.com/jiangfeng1124/transfer)
- Multi-Domain Adversarial Learning [[ICLR2019]](https://openreview.net/forum?id=Sklv5iRqYX)
- Moment Matching for Multi-Source Domain Adaptation [[arXiv 4 Dec 2018]](https://arxiv.org/abs/1812.01754v1)
- Algorithms and Theory for Multiple-Source Adaptation [[NIPS2018]](https://papers.nips.cc/paper/8046-algorithms-and-theory-for-multiple-source-adaptation)
- Adversarial Multiple Source Domain Adaptation [[NIPS2018]](http://papers.nips.cc/paper/8075-adversarial-multiple-source-domain-adaptation)
- Deep Cocktail Network: Multi-source Unsupervised Domain Adaptation with Category Shift [[CVPR2018]](https://arxiv.org/abs/1803.00830) [[Pytorch]](https://github.com/HCPLab-SYSU/MSDA)
- A survey of multi-source domain adaptation [[Information Fusion]](https://www.sciencedirect.com/science/article/pii/S1566253514001316)

## Multi Target DA
- Unsupervised Multi-Target Domain Adaptation: An Information Theoretic Approach [[arXiv]](https://arxiv.org/abs/1810.11547v1)


## Multi Step DA
- Distant domain transfer learning [[AAAI2017]](http://www.ntu.edu.sg/home/sinnopan/publications/[AAAI17]Distant%20Domain%20Transfer%20Learning.pdf)

## General Transfer Learning
### Domain Generalization
- Domain Generalization by Solving Jigsaw Puzzles [[CVPR2019]](https://arxiv.org/abs/1903.06864)
- Beyond Domain Adaptation: Unseen Domain Encapsulation via Universal Non-volume Preserving Models [[arXiv 9 Dec 2018]](https://arxiv.org/abs/1812.03407v1)
- Domain Generalization with Adversarial Feature Learning [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Domain_Generalization_With_CVPR_2018_paper.pdf)
- Deep Domain Generalization via Conditional Invariant Adversarial Networks [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.pdf)
- MetaReg: Towards Domain Generalization using Meta-Regularization [[NIPS2018]](https://papers.nips.cc/paper/7378-metareg-towards-domain-generalization-using-meta-regularization)

### Meta-Learning
Unsupervised Learning via Meta-Learning [[arXiv]](https://arxiv.org/abs/1810.02334)

### Transfer Metric Learning
- Transfer Metric Learning: Algorithms, Applications and Outlooks [[arXiv]](https://arxiv.org/abs/1810.03944)

### Others
- When Semi-Supervised Learning Meets Transfer Learning: Training Strategies, Models and Datasets [[arXiv 13 Dec 2018]](https://arxiv.org/abs/1812.05313)

## Applications
### Object Detection
- Cross-Domain Weakly-Supervised Object Detection Through Progressive Domain Adaptation [[CVPR2018]](https://arxiv.org/abs/1803.11365)
- Domain Adaptive Faster R-CNN for Object Detection in the Wild [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Domain_Adaptive_Faster_CVPR_2018_paper.pdf)

### Semantic Segmentation
- Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation [[CVPR2019]](https://arxiv.org/abs/1809.09478)
- SPIGAN: Privileged Adversarial Learning from Simulation [[ICLR2019]](https://openreview.net/forum?id=rkxoNnC5FQ)
- ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation [[arXiv 30 Nov 2018]](https://arxiv.org/abs/1811.12833v1)
- Unsupervised domain adaptation for medical imaging segmentation with self-ensembling [[NIPS2018]](https://arxiv.org/abs/1811.06042v1)
- Domain transfer through deep activation matching [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Haoshuo_Huang_Domain_transfer_through_ECCV_2018_paper.pdf)
- Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yang_Zou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.pdf)
- Conditional Generative Adversarial Network for Structured Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hong_Conditional_Generative_Adversarial_CVPR_2018_paper.pdf)
- Learning From Synthetic Data: Addressing Domain Shift for Semantic Segmentation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sankaranarayanan_Learning_From_Synthetic_CVPR_2018_paper.pdf)
- Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes [[ICCV2017]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Curriculum_Domain_Adaptation_ICCV_2017_paper.pdf) [[Journal Version]](https://arxiv.org/abs/1812.09953v3)

### Person Re-identification
- EANet: Enhancing Alignment for Cross-Domain Person Re-identification [[arXiv 29 Dec 2018]](https://arxiv.org/abs/1812.11369) [[Pytorch]](https://github.com/huanghoujing/EANet)
- One Shot Domain Adaptation for Person Re-Identification [[arXiv 26 Nov 2018]](https://arxiv.org/abs/1811.10144v1)
- Similarity-preserving Image-image Domain Adaptation for Person Re-identification [[arXiv 26 Nov 2018]](https://arxiv.org/abs/1811.10551v1)
- Domain Adaptation through Synthesis for Unsupervised Person Re-identification [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Slawomir_Bak_Domain_Adaptation_through_ECCV_2018_paper.pdf)
- Person Transfer GAN to Bridge Domain Gap for Person Re-Identification [[CVPR2018]](https://arxiv.org/abs/1711.08565v2) 
- Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification [[CVPR2018]](https://arxiv.org/abs/1711.07027v3)

### Medical Related
- Synergistic Image and Feature Adaptation: Towards Cross-Modality Domain Adaptation for Medical Image Segmentation [[arXiv on 24 Jan 2019]](https://arxiv.org/abs/1901.08211v1)
- Unsupervised domain adaptation for medical imaging segmentation with self-ensembling [[arXiv 14 Nov 2018]](https://arxiv.org/abs/1811.06042v1)

### Others
- Real-Time Monocular Depth Estimation using Synthetic Data with Domain Adaptation via Image Style Transfer [[CVPR2018]](http://breckon.eu/toby/publications/papers/abarghouei18monocular.pdf)

## Benchmarks
- Syn2Real: A New Benchmark forSynthetic-to-Real Visual Domain Adaptation [[arXiv 26 Jun]](https://arxiv.org/abs/1806.09755v1) [[Project]](http://ai.bu.edu/syn2real/)

# Library
- [Xlearn:Transfer Learning Library](https://github.com/thuml/Xlearn)
- [deep-transfer-learning:a PyTorch library for deep transfer learning](https://github.com/easezyc/deep-transfer-learning)
- [salad:a Semi-supervised Adaptive Learning Across Domains](https://domainadaptation.org/)

# Other Resources
- [transferlearning](https://github.com/jindongwang/transferlearning)
