# awsome-domain-adaptation

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 

This repo is a collection of AWESOME things about domain adaptation, including papers, code, etc. Feel free to star and fork.

# Contents
- [awsome-domain-adaptation](#awsome-domain-adaptation)
- [Contents](#contents)
- [Papers](#papers)
  - [Survey](#survey)
  - [Theory](#theory)
  - [Unsupervised DA](#unsupervised-da)
    - [Adversarial Methods](#adversarial-methods)
    - [Distance-based Methods](#distance-based-methods)
    - [Optimal Transport](#optimal-transport)
    - [Incremental Methods](#incremental-methods)
    - [Other Methods](#other-methods)
  - [Semi-supervised DA](#semi-supervised-da)
  - [Weakly-Supervised DA](#weakly-supervised-da)
  - [Zero-shot DA](#zero-shot-da)
  - [One-shot DA](#one-shot-da)
  - [Few-shot DA](#few-shot-da)
  - [Open Set DA](#open-set-da)
  - [Partial DA](#partial-da)
  - [Universal DA](#universal-da)
  - [Multi Source DA](#multi-source-da)
  - [Multi Target DA](#multi-target-da)
  - [Multi Step DA](#multi-step-da)
  - [Heterogeneous DA](#heterogeneous-da)
  - [Target-agnostic DA](#target-agnostic-da)
  - [Federated DA](#federated-da)
  - [Model Selection](#model-selection)
  - [Other Transfer Learning Paradigms](#other-transfer-learning-paradigms)
    - [Domain Generalization](#domain-generalization)
    - [Domain Randomization](#domain-randomization)
    - [Transfer Metric Learning](#transfer-metric-learning)
    - [Knowledge Transfer](#knowledge-transfer)
    - [Others](#others)
  - [Applications](#applications)
    - [Object Detection](#object-detection)
    - [Semantic Segmentation](#semantic-segmentation)
    - [Person Re-identification](#person-re-identification)
    - [Video Domain Adaptation](#video-domain-adaptation)
    - [Medical Related](#medical-related)
    - [Monocular Depth Estimation](#monocular-depth-estimation)
    - [3D Reconstruction](#3d-reconstruction)
    - [Others](#others-1)
  - [Related Topics](#related-topics)
    - [Image-to-Image Translation](#image-to-image-translation)
    - [Disentangled Representation Learning](#disentangled-representation-learning)
  - [Benchmarks](#benchmarks)
- [Library](#library)
- [Lectures and Tutorials](#lectures-and-tutorials)
- [Other Resources](#other-resources)

# Papers
## Survey
**Arxiv**
- A Comprehensive Survey on Transfer Learning [[7 Nov 2019]](https://arxiv.org/abs/1911.02685)
- Transfer Adaptation Learning: A Decade Survey [[12 Mar 2019]](https://arxiv.org/abs/1903.04687)
- A review of single-source unsupervised domain adaptation [[16 Jan 2019]](https://arxiv.org/abs/1901.05335)
- An introduction to domain adaptation and transfer learning [[31 Dec 2018]](https://arxiv.org/abs/1812.11806v2)
- A Survey of Unsupervised Deep Domain Adaptation [[6 Dec 2018]](https://arxiv.org/abs/1812.02849v2)
- Transfer Learning for Cross-Dataset Recognition: A Survey [[2017]](https://sci-hub.tw/https://arxiv.org/abs/1705.04396)
- Domain Adaptation for Visual Applications: A Comprehensive Survey  [[2017]](https://arxiv.org/abs/1702.05374)

**Journal**
- Deep Visual Domain Adaptation: A Survey [[Neurocomputing 2018]](https://arxiv.org/abs/1802.03601v4)
- A Survey on Deep Transfer Learning [[ICANN2018]](https://arxiv.org/abs/1808.01974v1)
- Visual domain adaptation: A survey of recent advances [[2015]](https://sci-hub.tw/10.1109/msp.2014.2347059)

## Theory
**Arxiv**
- Learning Bounds for Moment-Based Domain Adaptation [[19 Feb 2020]](https://arxiv.org/abs/2002.08260)
- A General Upper Bound for Unsupervised Domain Adaptation [[3 Oct 2019]](https://arxiv.org/abs/1910.01409)
- On Deep Domain Adaptation: Some Theoretical Understandings [[arXiv 15 Nov 2018]](https://arxiv.org/abs/1811.06199)
- Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift [[arXiv 10 Mar 2020]](https://arxiv.org/abs/2003.04475)

**Conference**
- Bridging Theory and Algorithm for Domain Adaptation [[ICML2019]](http://proceedings.mlr.press/v97/zhang19i/zhang19i.pdf) [[Pytorch]](https://github.com/thuml/MDD)
- On Learning Invariant Representation for Domain Adaptation [[ICML2019]](https://arxiv.org/abs/1901.09453v1) [[code]](https://github.com/KeiraZhao/On-Learning-Invariant-Representations-for-Domain-Adaptation)
- Unsupervised Domain Adaptation Based on Source-guided Discrepancy [[AAAI2019]](https://arxiv.org/abs/1809.03839)
- Learning Bounds for Domain Adaptation [[NIPS2007]](http://papers.nips.cc/paper/3212-learning-bounds-for-domain-adaptation)
- Analysis of Representations for Domain Adaptation [[NIPS2006]](https://papers.nips.cc/paper/2983-analysis-of-representations-for-domain-adaptation)

**Journal**
- A theory of learning from different domains [[ML2010]](https://link.springer.com/content/pdf/10.1007%2Fs10994-009-5152-4.pdf)

## Unsupervised DA

### Adversarial Methods

**Conference**
- Adversarial-Learned Loss for Domain Adaptation [[AAAI2020]](https://arxiv.org/abs/2001.01046v1)
- Adversarial Domain Adaptation with Domain Mixup [[AAAI2020]](https://arxiv.org/abs/1912.01805v1) [[Pytorch]](https://github.com/ChrisAllenMing/Mixup_for_UDA)
- Discriminative Adversarial Domain Adaptation [[AAAI2020]](https://arxiv.org/abs/1911.12036v1) [[Pytorch]](https://github.com/huitangtang/DADA-AAAI2020)
- Unifying Unsupervised Domain Adaptation and Zero-Shot Visual Recognition [[IJCNN2019]](https://arxiv.org/abs/1903.10601) [[Matlab]](https://github.com/hellowangqian/domain-adaptation-capls)
- Transfer Learning with Dynamic Adversarial Adaptation Network [[ICDM2019]](https://arxiv.org/abs/1909.08184)
- Joint Adversarial Domain Adaptation [[ACM MM2019]](https://dl.acm.org/citation.cfm?id=3351070)
- Cycle-consistent Conditional Adversarial Transfer Networks [[ACM MM2019]](https://dl.acm.org/citation.cfm?id=3350902) [[Pytorch]](https://github.com/lijin118/3CATN)
- Learning Disentangled Semantic Representation for Domain Adaptation [[IJCAI2019]](https://www.ijcai.org/proceedings/2019/0285.pdf) [[Tensorflow]](https://github.com/DMIRLAB-Group/DSR)
- Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation [[ICML2019]](http://proceedings.mlr.press/v97/chen19i/chen19i.pdf) [[Pytorch]](https://github.com/thuml/Batch-Spectral-Penalization)
- Transferable Adversarial Training: A General Approach to Adapting Deep Classifiers [[ICML2019]](http://proceedings.mlr.press/v97/liu19b/liu19b.pdf) [[Pytorch]](https://github.com/thuml/Transferable-Adversarial-Training)
- Drop to Adapt: Learning Discriminative Features for Unsupervised Domain Adaptation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lee_Drop_to_Adapt_Learning_Discriminative_Features_for_Unsupervised_Domain_Adaptation_ICCV_2019_paper.pdf) [[PyTorch]](https://github.com/postBG/DTA.pytorch)
- Cluster Alignment with a Teacher for Unsupervised Domain Adaptation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Deng_Cluster_Alignment_With_a_Teacher_for_Unsupervised_Domain_Adaptation_ICCV_2019_paper.pdf) [[Tensorflow]](https://github.com/thudzj/CAT)
- Unsupervised Domain Adaptation via Regularized Conditional Alignment [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cicek_Unsupervised_Domain_Adaptation_via_Regularized_Conditional_Alignment_ICCV_2019_paper.pdf)
- Attending to Discriminative Certainty for Domain Adaptation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kurmi_Attending_to_Discriminative_Certainty_for_Domain_Adaptation_CVPR_2019_paper.pdf) [[Project]](https://delta-lab-iitk.github.io/CADA/)
- GCAN: Graph Convolutional Adversarial Network for Unsupervised Domain Adaptation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ma_GCAN_Graph_Convolutional_Adversarial_Network_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)
- Domain-Symmetric Networks for Adversarial Domain Adaptation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Domain-Symmetric_Networks_for_Adversarial_Domain_Adaptation_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/YBZh/SymNets)
- DLOW: Domain Flow for Adaptation and Generalization [[CVPR2019 Oral]](https://arxiv.org/pdf/1812.05418.pdf)
- Progressive Feature Alignment for Unsupervised Domain Adaptation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Progressive_Feature_Alignment_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf) [[Tensorflow]](https://github.com/Xiewp/PFAN)
- Gotta Adapt ’Em All: Joint Pixel and Feature-Level Domain Adaptation for Recognition in the Wild [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tran_Gotta_Adapt_Em_All_Joint_Pixel_and_Feature-Level_Domain_Adaptation_CVPR_2019_paper.pdf) 
- Looking back at Labels: A Class based Domain Adaptation Technique [[IJCNN2019]](https://arxiv.org/abs/1904.01341) [[Project]](https://vinodkkurmi.github.io/DiscriminatorDomainAdaptation/)
- Consensus Adversarial Domain Adaptation [[AAAI2019]](https://aaai.org/ojs/index.php/AAAI/article/view/4552)
- Transferable Attention for Domain Adaptation [[AAAI2019]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transferable-attention-aaai19.pdf)
- Exploiting Local Feature Patterns for Unsupervised Domain Adaptation [[AAAI2019]](https://arxiv.org/abs/1811.05042v2)
- Augmented Cyclic Adversarial Learning for Low Resource Domain Adaptation [[ICLR2019]](https://openreview.net/forum?id=B1G9doA9F7)
- Conditional Adversarial Domain Adaptation [[NIPS2018]](http://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation) [[Pytorch(official)]](https://github.com/thuml/CDAN)  [[Pytorch(third party)]](https://github.com/thuml/CDAN)
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
- Wasserstein Distance Guided Representation Learning for Domain Adaptation [[AAAI2018]](https://arxiv.org/abs/1707.01217) [[TensorFlow(official)]](https://github.com/RockySJ/WDGRL) [[Pytorch]](https://github.com/jvanvugt/pytorch-domain-adaptation)
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

**Journal**
- Adversarial Learning and Interpolation Consistency for Unsupervised Domain Adaptation [[IEEE ACCESS]](https://ieeexplore.ieee.org/document/8913529)
- TarGAN: Generating target data with class labels for unsupervised domain adaptation [[Knowledge-Based Systems]]()

**Arxiv**
- Bi-Directional Generation for Unsupervised Domain Adaptation [[12 Feb 2020]](https://arxiv.org/abs/2002.04869v1)
- Enlarging Discriminative Power by Adding an Extra Class in Unsupervised Domain Adaptation [[19 Feb 2020]](https://arxiv.org/abs/2002.08041v1) [[Tensorflow]](https://github.com/haitran14/gada)
- Learning Domain Adaptive Features with Unlabeled Domain Bridges [[10 Dec 2019]](https://arxiv.org/abs/1912.05004v1)
- Reducing Domain Gap via Style-Agnostic Networks [[25 Oct 2019]](https://arxiv.org/abs/1910.11645)
- Generalized Domain Adaptation with Covariate and
Label Shift CO-ALignment [[23 Oct 2019]](https://arxiv.org/abs/1910.10320)
- Adversarial Variational Domain Adaptation [[25 Sep 2019]](https://arxiv.org/abs/1909.11651)
- Contrastively Smoothed Class Alignment for Unsupervised Domain Adaptation [[arXiv 13 Sep 2019]](https://arxiv.org/abs/1909.05288)
- SALT: Subspace Alignment as an Auxiliary Learning Task for Domain Adaptation [[arXiv 11 Jun 2019]](https://arxiv.org/abs/1906.04338v1)
- Joint Semantic Domain Alignment and Target Classifier Learning for Unsupervised Domain Adaptation [[arXiv 10 Jun 2019]](https://arxiv.org/abs/1906.04053v1)
- Adversarial Domain Adaptation Being Aware of Class Relationships [[arXiv 28 May 2019]](https://arxiv.org/abs/1905.11931v1)
- Domain-Invariant Adversarial Learning for Unsupervised Domain Adaption [[arXiv 30 Nov 2018]](https://arxiv.org/abs/1811.12751)
- Unsupervised Domain Adaptation using Deep Networks with Cross-Grafted Stacks [[arXiv 17 Feb 2019]](https://arxiv.org/abs/1902.06328v1)
- DART: Domain-Adversarial Residual-Transfer Networks for Unsupervised Cross-Domain Image Classification [[arXiv 30 Dec 2018]](https://arxiv.org/abs/1812.11478)
- Unsupervised Domain Adaptation using Generative Models and Self-ensembling [[arXiv 2 Dec 2018]](https://arxiv.org/abs/1812.00479)
- Domain Confusion with Self Ensembling for Unsupervised Adaptation [[arXiv 10 Oct 2018]](https://arxiv.org/abs/1810.04472)
- Improving Adversarial Discriminative Domain Adaptation [[arXiv 10 Sep 2018]](https://arxiv.org/abs/1809.03625)
- M-ADDA: Unsupervised Domain Adaptation with Deep Metric Learning [[arXiv 6 Jul 2018]](https://arxiv.org/abs/1807.02552v1) [[Pytorch(official)]](https://github.com/IssamLaradji/M-ADDA)
- Factorized Adversarial Networks for Unsupervised Domain Adaptation [[arXiv 4 Jun 2018]](https://arxiv.org/abs/1806.01376v1)
- DiDA: Disentangled Synthesis for Domain Adaptation [[arXiv 21 May 2018]](https://arxiv.org/abs/1805.08019v1)
- Unsupervised Domain Adaptation with Adversarial Residual Transform Networks [[arXiv 25 Apr 2018]](https://arxiv.org/abs/1804.09578)
- Causal Generative Domain Adaptation Networks [[arXiv 28 Jun 2018]](https://arxiv.org/abs/1804.04333v3)


### Distance-based Methods
**Journal**

- Transferable Representation Learning with Deep Adaptation Networks [[TPAMI]](https://ieeexplore.ieee.org/document/8454781)

**Conference**
- HoMM: Higher-order Moment Matching for Unsupervised Domain Adaptation [[AAAI2020]](https://arxiv.org/abs/1912.11976) [[Tensorflow]](https://github.com/chenchao666/HoMM-Master)
- Normalized Wasserstein for Mixture Distributions With Applications in Adversarial Learning and Domain Adaptation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Balaji_Normalized_Wasserstein_for_Mixture_Distributions_With_Applications_in_Adversarial_Learning_ICCV_2019_paper.pdf)
- Joint Domain Alignment and Discriminative Feature Learning for Unsupervised Deep Domain Adaptation [[AAAI2019]](https://arxiv.org/abs/1808.09347v2)
- Residual Parameter Transfer for Deep Domain Adaptation [[CVPR2018]](https://arxiv.org/abs/1711.07714)
- Deep Asymmetric Transfer Network for Unbalanced Domain Adaptation [[AAAI2018]](http://media.cs.tsinghua.edu.cn/~multimedia/cuipeng/papers/DATN.pdf)
- Central Moment Discrepancy for Unsupervised Domain Adaptation [[ICLR2017]](https://openreview.net/pdf?id=SkB-_mcel), [[InfSc2019]](https://arxiv.org/pdf/1711.06114.pdf), [[code]](https://github.com/wzell/cmd)
- Deep CORAL: Correlation Alignment for Deep Domain Adaptation [[ECCV2016]](https://arxiv.org/abs/1607.01719)
- Learning Transferable Features with Deep Adaptation Networks [[ICML2015]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-adaptation-networks-icml15.pdf)[[code]](https://github.com/thuml/DAN)
- Unsupervised Domain Adaptation with Residual Transfer Networks [[NIPS2016]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/residual-transfer-network-nips16.pdf) [[code]](https://github.com/thuml/Xlearn)
- Deep Transfer Learning with Joint Adaptation Networks [[ICML2017]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/joint-adaptation-networks-icml17.pdf) [[code]](https://github.com/thuml/Xlearn)

**Arxiv**
- Deep Domain Confusion: Maximizing for Domain Invariance [[Arxiv 2014]](https://arxiv.org/abs/1412.3474)


### Optimal Transport
- CDOT: Continuous Domain Adaptation using Optimal Transport [[20 Sep 2019]](https://arxiv.org/abs/1909.11448)
- Differentially Private Optimal Transport: Application to Domain Adaptation [[IJCAI]](https://www.ijcai.org/proceedings/2019/0395.pdf)
- DeepJDOT: Deep Joint distribution optimal transport for unsupervised domain adaptation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bharath_Bhushan_Damodaran_DeepJDOT_Deep_Joint_ECCV_2018_paper.pdf) [[Keras]](https://github.com/bbdamodaran/deepJDOT)
- Joint Distribution Optimal Transportation for Domain Adaptation [[NIPS2017]](http://papers.nips.cc/paper/6963-joint-distribution-optimal-transportation-for-domain-adaptation.pdf) [[python]](https://github.com/rflamary/JDOT) [[Python Optimal Transport Library]](https://github.com/rflamary/POT)

### Incremental Methods
- Incremental Adversarial Domain Adaptation for Continually Changing Environments [[ICRA2018]](https://arxiv.org/abs/1712.07436)
- Continuous Manifold based Adaptation for Evolving Visual Domains [[CVPR2014]](https://people.eecs.berkeley.edu/~jhoffman/papers/Hoffman_CVPR2014.pdf)

### Other Methods
**Arxiv**
- Partially-Shared Variational Auto-encoders for Unsupervised Domain Adaptation with Target Shift [[25 Jan 2020]](https://arxiv.org/abs/2001.07895v3)
- Less Confusion More Transferable: Minimum Class Confusion for Versatile Domain Adaptation [[8 Dec 2019]](https://arxiv.org/abs/1912.03699v1)
- Improving Unsupervised Domain Adaptation with Variational Information Bottleneck [[21 Nov 2019]](https://arxiv.org/abs/1911.09310v1)
- Deep causal representation learning for unsupervised domain adaptation [[28 Oct 2019]](https://arxiv.org/abs/1910.12417)
- Domain-invariant Learning using Adaptive Filter
Decomposition [[25 Sep 2019]](https://arxiv.org/abs/1909.11285)
- Discriminative Clustering for Robust Unsupervised Domain Adaptation [[arXiv 30 May 2019]](https://arxiv.org/abs/1905.13331)
- Virtual Mixup Training for Unsupervised Domain Adaptation [[arXiv on 24 May 2019]](https://arxiv.org/abs/1905.04215) [[Tensorflow]](https://github.com/xudonmao/VMT)
- Learning Smooth Representation for Unsupervised Domain Adaptation [[arXiv 26 May 2019]](https://arxiv.org/abs/1905.10748v1)
- Towards Self-similarity Consistency and Feature Discrimination for Unsupervised Domain Adaptation [[arXiv 13 Apr 2019]](https://arxiv.org/abs/1904.06490v1)
- Easy Transfer Learning By Exploiting Intra-domain Structures [[arXiv 2 Apr 2019]](https://arxiv.org/abs/1904.01376v1) 
- Domain Discrepancy Measure Using Complex Models in Unsupervised Domain Adaptation [[arXiv 30 Jan 2019]](https://arxiv.org/abs/1901.10654v1)
- Domain Alignment with Triplets [[arXiv 22 Jan 2019]](https://arxiv.org/abs/1812.00893v2)
- Deep Discriminative Learning for Unsupervised Domain Adaptation [[arXiv 17 Nov 2018]](https://arxiv.org/abs/1811.07134v1)

**Conference**
- Domain Adaptive Multiflow Networks [[ICLR2020]](https://openreview.net/forum?id=rJxycxHKDS)
- Unsupervised Domain Adaptation via Discriminative Manifold Embedding and Alignment [[AAAI2020]](https://arxiv.org/abs/2002.08675v1)
- Unsupervised Domain Adaptation via Structured Prediction Based Selective Pseudo-Labeling [[AAAI2020]](https://arxiv.org/abs/1911.07982) [[Matlab]](https://github.com/hellowangqian/domain-adaptation-capls)
- CUDA: Contradistinguisher for Unsupervised Domain Adaptation [[ICDM2019]](https://arxiv.org/abs/1909.03442)
- Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment [[ICML2019]](http://proceedings.mlr.press/v97/wu19f/wu19f.pdf)
- Batch Weight for Domain Adaptation With Mass Shift [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Binkowski_Batch_Weight_for_Domain_Adaptation_With_Mass_Shift_ICCV_2019_paper.pdf)
- Switchable Whitening for Deep Representation Learning [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Pan_Switchable_Whitening_for_Deep_Representation_Learning_ICCV_2019_paper.pdf) [[pytorch]](https://github.com/XingangPan/Switchable-Whitening)
- Confidence Regularized Self-Training [[ICCV2019 Oral]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zou_Confidence_Regularized_Self-Training_ICCV_2019_paper.pdf) [[Pytorch]](https://github.com/yzou2/CRST)
- Larger Norm More Transferable: An Adaptive Feature Norm Approach for Unsupervised Domain Adaptation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Larger_Norm_More_Transferable_An_Adaptive_Feature_Norm_Approach_for_ICCV_2019_paper.pdf) [[Pytorch(official)]](https://github.com/jihanyang/AFN)
- Transferrable Prototypical Networks for Unsupervised Domain Adaptation [[CVPR2019(Oral)]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pan_Transferrable_Prototypical_Networks_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)
- Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lee_Sliced_Wasserstein_Discrepancy_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)
- Unsupervised Domain Adaptation using Feature-Whitening and Consensus Loss [[CVPR 2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Roy_Unsupervised_Domain_Adaptation_Using_Feature-Whitening_and_Consensus_Loss_CVPR_2019_paper.pdf)  [[Pytorch]](https://github.com/roysubhankar/dwt-domain-adaptation)
- Domain Specific Batch Normalization for Unsupervised Domain Adaptation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_Domain-Specific_Batch_Normalization_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)
- AdaGraph: Unifying Predictive and Continuous Domain Adaptation through Graphs [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Mancini_AdaGraph_Unifying_Predictive_and_Continuous_Domain_Adaptation_Through_Graphs_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/mancinimassimiliano/adagraph)
- Unsupervised Visual Domain Adaptation: A Deep Max-Margin Gaussian Process Approach [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Unsupervised_Visual_Domain_Adaptation_A_Deep_Max-Margin_Gaussian_Process_Approach_CVPR_2019_paper.pdf) [[Project]](https://seqam-lab.github.io/GPDA/)
- Contrastive Adaptation Network for Unsupervised Domain Adaptation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kang_Contrastive_Adaptation_Network_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/kgl-prml/Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation)
- Distant Supervised Centroid Shift: A Simple and Efficient Approach to Visual Domain Adaptation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liang_Distant_Supervised_Centroid_Shift_A_Simple_and_Efficient_Approach_to_CVPR_2019_paper.pdf)
- Unsupervised Domain Adaptation via Calibrating Uncertainties [[CVPRW2019]](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Han_Unsupervised_Domain_Adaptation_via_Calibrating_Uncertainties_CVPRW_2019_paper.pdf)
- Bayesian Uncertainty Matching for Unsupervised Domain Adaptation [[IJCAI2019]](https://arxiv.org/abs/1906.09693v1)
- Unsupervised Domain Adaptation for Distance Metric Learning [[ICLR2019]](https://openreview.net/forum?id=BklhAj09K7)
- Co-regularized Alignment for Unsupervised Domain Adaptation [[NIPS2018]](http://papers.nips.cc/paper/8146-co-regularized-alignment-for-unsupervised-domain-adaptation)
- Domain Invariant and Class Discriminative Feature Learning for Visual Domain Adaptation [[TIP 2018]](https://ieeexplore.ieee.org/document/8362753/)
- Graph Adaptive Knowledge Transfer for Unsupervised Domain Adaptation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhengming_Ding_Graph_Adaptive_Knowledge_ECCV_2018_paper.pdf)
- Aligning Infinite-Dimensional Covariance Matrices in Reproducing Kernel Hilbert Spaces for Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Aligning_Infinite-Dimensional_Covariance_CVPR_2018_paper.pdf)
- Unsupervised Domain Adaptation with Distribution Matching Machines [[AAAI2018]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/distribution-matching-machines-aaai18.pdf)
- Learning to cluster in order to transfer across domains and tasks [[ICLR2018]](https://openreview.net/forum?id=ByRWCqvT-) [[Bolg]](https://mlatgt.blog/2018/04/29/learning-to-cluster/) [[Pytorch]](https://github.com/GT-RIPL/L2C)
- Self-Ensembling for Visual Domain Adaptation [[ICLR2018]](https://openreview.net/forum?id=rkpoTaxA-)
- Minimal-Entropy Correlation Alignment for Unsupervised Deep Domain Adaptation [[ICLR2018]](https://openreview.net/forum?id=rJWechg0Z) [[TensorFlow]](https://github.com/pmorerio/minimal-entropy-correlation-alignment)
- Associative Domain Adaptation [[ICCV2017]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Haeusser_Associative_Domain_Adaptation_ICCV_2017_paper.pdf) [[TensorFlow]](https://github.com/haeusser/learning_by_association) [[Pytorch]](https://github.com/corenel/pytorch-atda)
- AutoDIAL: Automatic DomaIn Alignment Layers [[ICCV2017]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Carlucci_AutoDIAL_Automatic_DomaIn_ICCV_2017_paper.pdf)
- Asymmetric Tri-training for Unsupervised Domain Adaptation [[ICML2017]](http://proceedings.mlr.press/v70/saito17a.html) [[TensorFlow]](https://github.com/ksaito-ut/atda)
- Learning Transferrable Representations for Unsupervised Domain Adaptation [[NIPS2016]](http://papers.nips.cc/paper/6360-learning-transferrable-representations-for-unsupervised-domain-adaptation)

**Journal**
- Adaptive Batch Normalization for practical domain adaptation [[Pattern Recognition(2018)]](https://www.sciencedirect.com/science/article/pii/S003132031830092X)
- Unsupervised Domain Adaptation by Mapped Correlation Alignment [[IEEE ACCESS]](https://ieeexplore.ieee.org/abstract/document/8434290/)

## Semi-supervised DA
**Arxiv**
- Opposite Structure Learning for Semi-supervised Domain Adaptation [[6 Feb 2020]](https://arxiv.org/abs/2002.02545v1)
- Reducing Domain Gap via Style-Agnostic Networks [[25 Oct 2019]](https://arxiv.org/abs/1910.11645)

**Conference**
- Semi-supervised Domain Adaptation via Minimax Entropy [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Saito_Semi-Supervised_Domain_Adaptation_via_Minimax_Entropy_ICCV_2019_paper.pdf) [[Pytorch]](https://github.com/VisionLearningGroup/SSDA_MME)

## Weakly-Supervised DA
**Arxiv**
- Butterfly: Robust One-step Approach towards Wildly-unsupervised Domain Adaptation [[arXiv on 19 May 2019]](https://arxiv.org/abs/1905.07720v1)

**Conference**
- Weakly Supervised Open-set Domain Adaptation by Dual-domain Collaboration [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_Weakly_Supervised_Open-Set_Domain_Adaptation_by_Dual-Domain_Collaboration_CVPR_2019_paper.pdf)
- Transferable Curriculum for Weakly-Supervised Domain Adaptation [[AAAI2019]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transferable-curriculum-aaai19.pdf)

## Zero-shot DA
**Arxiv**


**Conference**

- Zero-shot Domain Adaptation Based on Attribute Information [[ACML2019]](http://proceedings.mlr.press/v101/ishii19a.html)
- Conditional Coupled Generative Adversarial Networks for Zero-Shot Domain Adaptation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Conditional_Coupled_Generative_Adversarial_Networks_for_Zero-Shot_Domain_Adaptation_ICCV_2019_paper.pdf)
- Generalized Zero-Shot Learning with Deep Calibration Network [NIPS2018](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-calibration-network-nips18.pdf)
- Zero-Shot Deep Domain Adaptation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kuan-Chuan_Peng_Zero-Shot_Deep_Domain_ECCV_2018_paper.pdf)

## One-shot DA
- One-Shot Imitation from Observing Humans via Domain-Adaptive Meta-Learning [[arxiv]](https://arxiv.org/abs/1802.01557)
- One-Shot Adaptation of Supervised Deep Convolutional Models [[ICLR Workshop 2014]](https://arxiv.org/abs/1312.6204)

## Few-shot DA
**Conference**
- Few-Shot Adaptive Faster R-CNN [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Few-Shot_Adaptive_Faster_R-CNN_CVPR_2019_paper.html)
- d-SNE: Domain Adaptation using Stochastic Neighborhood Embedding [[CVPR2019 Oral]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_d-SNE_Domain_Adaptation_Using_Stochastic_Neighborhood_Embedding_CVPR_2019_paper.pdf)
- Few-Shot Adversarial Domain Adaptation [[NIPS2017]](http://papers.nips.cc/paper/7244-few-shot-adversarial-domain-adaptation)

**Arxiv**
- Few-shot Domain Adaptation by Causal Mechanism Transfer [[10 Feb 2020]](https://arxiv.org/abs/2002.03497v1)

## Open Set DA

**Conference**
- Separate to Adapt: Open Set Domain Adaptation via Progressive Separation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Separate_to_Adapt_Open_Set_Domain_Adaptation_via_Progressive_Separation_CVPR_2019_paper.pdf)
- Weakly Supervised Open-set Domain Adaptation by Dual-domain Collaboration [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_Weakly_Supervised_Open-Set_Domain_Adaptation_by_Dual-Domain_Collaboration_CVPR_2019_paper.pdf)
- Learning Factorized Representations for Open-set Domain Adaptation [[ICLR2019]](https://openreview.net/pdf?id=SJe3HiC5KX)
- Open Set Domain Adaptation by Backpropagation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kuniaki_Saito_Adversarial_Open_Set_ECCV_2018_paper.pdf) [[Pytorch(Official)]](https://github.com/ksaito-ut/OPDA_BP) [[Tensorflow]](https://github.com/Mid-Push/Open_set_domain_adaptation) [[Pytorch]](https://github.com/YU1ut/openset-DA)
- Open Set Domain Adaptation [[ICCV2017]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Busto_Open_Set_Domain_ICCV_2017_paper.pdf)

**Arxiv**
- Collaborative Training of Balanced Random Forests for Open Set Domain Adaptation [[10 Feb 2020]](https://arxiv.org/abs/2002.03642v1)
- Known-class Aware Self-ensemble for Open Set Domain Adaptation [[3 May 2019]](https://arxiv.org/abs/1905.01068v1)

## Partial DA
**Arxiv**
- Tackling Partial Domain Adaptation with Self-Supervision [[arXiv 12 Jun 2019]](https://arxiv.org/abs/1906.05199v1)
- Selective Transfer with Reinforced Transfer Network for Partial Domain Adaptation [[arXiv 26 May 2019]](https://arxiv.org/abs/1905.10756v1)
- Domain Adversarial Reinforcement Learning for Partial Domain Adaptation [[arXiv 10 May 2019]](https://arxiv.org/abs/1905.04094v1)

**Conference**
- Learning to Transfer Examples for Partial Domain Adaptation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cao_Learning_to_Transfer_Examples_for_Partial_Domain_Adaptation_CVPR_2019_paper.pdf)
- Partial Adversarial Domain Adaptation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhangjie_Cao_Partial_Adversarial_Domain_ECCV_2018_paper.pdf) [[Pytorch(Official)]](https://github.com/thuml/PADA)
- Importance Weighted Adversarial Nets for Partial Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Importance_Weighted_Adversarial_CVPR_2018_paper.html)
- Partial Transfer Learning with Selective Adversarial Networks [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cao_Partial_Transfer_Learning_CVPR_2018_paper.pdf)[[paper weekly]](http://www.paperweekly.site/papers/1388) [[Pytorch(Official) & Caffe(official)]](https://github.com/thuml/SAN)

## Universal DA
**Conference**
- Universal Domain Adaptation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/You_Universal_Domain_Adaptation_CVPR_2019_paper.pdf)  [[Pytorch]](https://github.com/thuml/Universal-Domain-Adaptation)

**Arxiv**
- Universal Domain Adaptation through Self Supervision [[19 Feb 2020]](http://arxiv.org/abs/2002.07953v1)
- A Sample Selection Approach for Universal Domain Adaptation [[14 Jan 2020]](https://arxiv.org/abs/2001.05071v1)

## Multi Source DA

**Conference**
- Multi-Source Domain Adaptation for Text Classification via DistanceNet-Bandits [[AAAI2020]](https://arxiv.org/abs/2001.04362v2)
- Multi-source Domain Adaptation for Visual Sentiment Classification [[AAAI2020]](https://arxiv.org/abs/2001.03886v1)
- Multi-source Distilling Domain Adaptation [[AAAI2020]](https://arxiv.org/abs/1911.11554v1) [[code]](https://github.com/daoyuan98/MDDA)
- Multi-source Domain Adaptation for Semantic Segmentation [[NeurlPS2019]](https://arxiv.org/abs/1910.12181) [[Pytorch]](https://github.com/Luodian/MADAN)
- Moment Matching for Multi-Source Domain Adaptation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Moment_Matching_for_Multi-Source_Domain_Adaptation_ICCV_2019_paper.pdf) [[Pytorch]](http://ai.bu.edu/M3SDA/)
- Multi-Domain Adversarial Learning [[ICLR2019]](https://openreview.net/forum?id=Sklv5iRqYX) [[Torch]](https://github.com/AltschulerWu-Lab/MuLANN)
- Algorithms and Theory for Multiple-Source Adaptation [[NIPS2018]](https://papers.nips.cc/paper/8046-algorithms-and-theory-for-multiple-source-adaptation)
- Adversarial Multiple Source Domain Adaptation [[NIPS2018]](http://papers.nips.cc/paper/8075-adversarial-multiple-source-domain-adaptation) [[Pytorch]](https://github.com/KeiraZhao/MDAN)
- Boosting Domain Adaptation by Discovering Latent Domains [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Mancini_Boosting_Domain_Adaptation_CVPR_2018_paper.pdf) [[Caffe]](https://github.com/mancinimassimiliano/latent_domains_DA) [[Pytorch]](https://github.com/mancinimassimiliano/pytorch_wbn)
- Deep Cocktail Network: Multi-source Unsupervised Domain Adaptation with Category Shift [[CVPR2018]](https://arxiv.org/abs/1803.00830) [[Pytorch]](https://github.com/HCPLab-SYSU/MSDA)

**Journal**
- A survey of multi-source domain adaptation [[Information Fusion]](https://www.sciencedirect.com/science/article/pii/S1566253514001316)

**Arxiv**
- Domain Adaptive Ensemble Learning [[arXiv]](https://arxiv.org/abs/2003.07325)
- Multi-Source Domain Adaptation and Semi-Supervised Domain Adaptation with Focus on Visual Domain Adaptation Challenge 2019 [[14 Oct 2019]](https://arxiv.org/abs/1910.03548)


## Multi Target DA
- Unsupervised Multi-Target Domain Adaptation: An Information Theoretic Approach [[arXiv]](https://arxiv.org/abs/1810.11547v1)


## Multi Step DA
**Arxiv**
- Adversarial Domain Adaptation for Stance Detection [[arXiv]](https://arxiv.org/abs/1902.02401)
- Ensemble Adversarial Training: Attacks and Defenses [[arXiv]](https://arxiv.org/abs/1705.07204)

**Conference**
- Distant domain transfer learning [[AAAI2017]](http://www.ntu.edu.sg/home/sinnopan/publications/[AAAI17]Distant%20Domain%20Transfer%20Learning.pdf)

## Heterogeneous DA
- Heterogeneous Domain Adaptation via Soft Transfer Network [[ACM MM2019]](https://arxiv.org/abs/1908.10552v1)

## Target-agnostic DA
**Arxiv**
- Compound Domain Adaptation in an Open World [[8 Sep 2019]](https://arxiv.org/abs/1909.03403)

**Conference**
- Domain Agnostic Learning with Disentangled Representations [[ICML2019]](http://proceedings.mlr.press/v97/peng19b/peng19b.pdf) [[Pytorch]](https://github.com/VisionLearningGroup/DAL)
- Blending-target Domain Adaptation by Adversarial Meta-Adaptation Networks [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Blending-Target_Domain_Adaptation_by_Adversarial_Meta-Adaptation_Networks_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/zjy526223908/BTDA)


## Federated DA
**Arxiv**
- Federated Adversarial Domain Adaptation [[5 Nov 2019]](https://arxiv.org/abs/1911.02054v1)

## Model Selection
- Towards Accurate Model Selection in Deep Unsupervised Domain Adaptation [[ICML2019]](http://proceedings.mlr.press/v97/you19a/you19a.pdf) [[Pytorch]](https://github.com/thuml/Deep-Embedded-Validation)

## Other Transfer Learning Paradigms
### Domain Generalization
**Arxiv**
- Adversarial Pyramid Network for Video Domain Generalization [[8 Dec 2019]](https://arxiv.org/abs/1912.03716)
- Towards Shape Biased Unsupervised Representation Learning for Domain Generalization [[18 Sep 2019]](https://arxiv.org/abs/1909.08245v1)
- A Generalization Error Bound for Multi-class Domain Generalization [[24 May 2019]](https://arxiv.org/abs/1905.10392v1)
- Adversarial Invariant Feature Learning with Accuracy Constraint for Domain Generalization [[29 Apr 2019]](https://arxiv.org/abs/1904.12543v1)
- Beyond Domain Adaptation: Unseen Domain Encapsulation via Universal Non-volume Preserving Models [[9 Dec 2018]](https://arxiv.org/abs/1812.03407v1)

**Conference**
- Generalized Convolutional Forest Networks for Domain Generalization and Visual Recognition [[ICLR2020]](https://openreview.net/forum?id=H1lxVyStPH)
- Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation [[ICLR2020]](https://openreview.net/forum?id=SJl5Np4tPr)
- Domain Generalization Using a Mixture of Multiple Latent Domains [[AAAI2020]](https://arxiv.org/abs/1911.07661v1) [[Pytorch]](https://github.com/mil-tokyo/dg_mmld)
- Domain Generalization via Model-Agnostic Learning of Semantic Features [[NeurIPS2019]](https://papers.nips.cc/paper/8873-domain-generalization-via-model-agnostic-learning-of-semantic-features) [[Tensorflow]](https://github.com/biomedia-mira/masf)
- Episodic Training for Domain Generalization [[ICCV2019 Oral]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Episodic_Training_for_Domain_Generalization_ICCV_2019_paper.pdf) [[code]](https://github.com/HAHA-DL/Episodic-DG)
- Feature-Critic Networks for Heterogeneous Domain Generalization [[ICML2019]](http://proceedings.mlr.press/v97/li19l/li19l.pdf) [[Pytorch]](https://github.com/liyiying/Feature_Critic)
- Domain Generalization by Solving Jigsaw Puzzles [[CVPR2019 Oral]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Carlucci_Domain_Generalization_by_Solving_Jigsaw_Puzzles_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/fmcarlucci/JigenDG)
- MetaReg: Towards Domain Generalization using Meta-Regularization [[NIPS2018]](https://papers.nips.cc/paper/7378-metareg-towards-domain-generalization-using-meta-regularization)
- Deep Domain Generalization via Conditional Invariant Adversarial Networks [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.pdf)
- Domain Generalization with Adversarial Feature Learning [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Domain_Generalization_With_CVPR_2018_paper.pdf)

**Journal**
- Correlation-aware Adversarial Domain Adaptation and Generalization [[Pattern Recognition(2019)]](https://arxiv.org/abs/1911.12983v1) [[code]](https://github.com/mahfujur1/CA-DA-DG)

### Domain Randomization
**Conference**
- DeceptionNet: Network-Driven Domain Randomization [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zakharov_DeceptionNet_Network-Driven_Domain_Randomization_ICCV_2019_paper.pdf)
- Domain Randomization and Pyramid Consistency: Simulation-to-Real Generalization Without Accessing Target Domain Data [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yue_Domain_Randomization_and_Pyramid_Consistency_Simulation-to-Real_Generalization_Without_Accessing_Target_ICCV_2019_paper.pdf)

### Transfer Metric Learning
- Transfer Metric Learning: Algorithms, Applications and Outlooks [[arXiv]](https://arxiv.org/abs/1810.03944)

### Knowledge Transfer
**Conference**
- Attention Bridging Network for Knowledge Transfer [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Attention_Bridging_Network_for_Knowledge_Transfer_ICCV_2019_paper.pdf)
- Few-Shot Image Recognition with Knowledge Transfer [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Few-Shot_Image_Recognition_With_Knowledge_Transfer_ICCV_2019_paper.pdf)


### Others
**Conference**
- Learning Across Tasks and Domains [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ramirez_Learning_Across_Tasks_and_Domains_ICCV_2019_paper.pdf)
- UM-Adapt: Unsupervised Multi-Task Adaptation Using Adversarial Cross-Task Distillation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Kundu_UM-Adapt_Unsupervised_Multi-Task_Adaptation_Using_Adversarial_Cross-Task_Distillation_ICCV_2019_paper.pdf)
- Domain Agnostic Learning with Disentangled Representations [[ICML2019]](https://arxiv.org/abs/1904.12347v1)
- Unsupervised Open Domain Recognition by Semantic Discrepancy Minimization [[CVPR2019]](https://arxiv.org/abs/1904.08631) [[Pytorch]](https://github.com/junbaoZHUO/UODTN)

**Arxiv**
- GradMix: Multi-source Transfer across Domains and Tasks [[9 Feb 2020]](GradMix: Multi-source Transfer across Domains and Tasks)
- When Semi-Supervised Learning Meets Transfer Learning: Training Strategies, Models and Datasets [[arXiv 13 Dec 2018]](https://arxiv.org/abs/1812.05313)


## Applications
### Object Detection
  
**Conference**
- Deep Domain Adaptive Object Detection: a Survey [[ICIP2020]](https://arxiv.org/abs/2002.06797v1)
- Progressive Domain Adaptation for Object Detection [[WACV]](https://arxiv.org/abs/1910.11319)
- Cross-Domain Car Detection Using Unsupervised Image-to-Image Translation: From Day to Night [[IJCNN2019 Oral]](https://ieeexplore.ieee.org/document/8852008) [[Project]](https://github.com/viniciusarruda/cross-domain-car-detection)
- Self-Training and Adversarial Background Regularization for Unsupervised Domain Adaptive One-Stage Object Detection [[ICCV2019 Oral]](https://arxiv.org/abs/1909.00597v1)
- A Robust Learning Approach to Domain Adaptive Object Detection [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Khodabandeh_A_Robust_Learning_Approach_to_Domain_Adaptive_Object_Detection_ICCV_2019_paper.pdf) [[code]](https://github.com/mkhodabandeh/robust_domain_adaptation)
- Multi-adversarial Faster-RCNN for Unrestricted Object Detection [[ICCV2019]](https://arxiv.org/abs/1907.10343)
- Few-Shot Adaptive Faster R-CNN [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Few-Shot_Adaptive_Faster_R-CNN_CVPR_2019_paper.html)
- Exploring Object Relation in Mean Teacher for Cross-Domain Detection [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cai_Exploring_Object_Relation_in_Mean_Teacher_for_Cross-Domain_Detection_CVPR_2019_paper.pdf)
- Adapting Object Detectors via Selective Cross-Domain Alignment [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Adapting_Object_Detectors_via_Selective_Cross-Domain_Alignment_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/xinge008/SCDA)
- Automatic adaptation of object detectors to new domains using self-training [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/RoyChowdhury_Automatic_Adaptation_of_Object_Detectors_to_New_Domains_Using_Self-Training_CVPR_2019_paper.pdf) [[Project]](http://vis-www.cs.umass.edu/unsupVideo/)
- Towards Universal Object Detection by Domain Attention [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Towards_Universal_Object_Detection_by_Domain_Attention_CVPR_2019_paper.pdf)
- Strong-Weak Distribution Alignment for Adaptive Object Detection [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Saito_Strong-Weak_Distribution_Alignment_for_Adaptive_Object_Detection_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/VisionLearningGroup/DA_Detection)
- Diversify and Match: A Domain Adaptive Representation Learning Paradigm for Object Detection [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Diversify_and_Match_A_Domain_Adaptive_Representation_Learning_Paradigm_for_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/TKKim93/DivMatch)
- Cross-Domain Weakly-Supervised Object Detection Through Progressive Domain Adaptation [[CVPR2018]](https://arxiv.org/abs/1803.11365)
- Domain Adaptive Faster R-CNN for Object Detection in the Wild [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Domain_Adaptive_Faster_CVPR_2018_paper.pdf) [[Caffe2]](https://github.com/krumo/Detectron-DA-Faster-RCNN) [[Caffe]](https://github.com/yuhuayc/da-faster-rcnn) [[Pytorch(under developing)]]()


**Journal**
- Pixel and feature level based domain adaptation for object detection in autonomous driving [[Neurocomputing]](https://www.sciencedirect.com/science/article/pii/S092523121931149X?via%3Dihub)

**Arxiv**
- Unsupervised Domain Adaptive Object Detection using Forward-Backward Cyclic Adaptation [[3 Feb 2020]](https://arxiv.org/abs/2002.00575v1)
- Prior-based Domain Adaptive Object Detection for Adverse Weather Conditions [[29 Nov 2019]](https://arxiv.org/abs/1912.00070v1)
- Unsupervised Domain Adaptation for Object Detection via Cross-Domain Semi-Supervised Learning [[17 Nov 2019]](https://arxiv.org/abs/1911.07158v1)
- Curriculum Self-Paced Learning for Cross-Domain Object Detection [[15 Nov 2019]](https://arxiv.org/abs/1911.06849v1)
- SCL: Towards Accurate Domain Adaptive Object Detection via Gradient Detach Based Stacked Complementary Losses [[6 Nov 2019]](https://arxiv.org/abs/1911.02559v1)

### Semantic Segmentation

**Conference**
- Unsupervised Instance Segmentation in Microscopy Images via Panoptic Domain Adaptation and Task Re-weighting [[CVPR2020]](https://arxiv.org/abs/2005.02066v1)
- Unsupervised Intra-domain Adaptation for Semantic Segmentation through Self-Supervision [[CVPR2020 Oral]](https://arxiv.org/abs/2004.07703v1) [[Pytorch]](https://github.com/feipan664/IntraDA)
- Differential Treatment for Stuff and Things: A Simple Unsupervised Domain Adaptation Method for Semantic Segmentation [[CVPR2020]](https://arxiv.org/abs/2003.08040v1)
- Learning Texture Invariant Representation for Domain Adaptation of Semantic Segmentation [[CVPR2020]](https://arxiv.org/abs/2003.00867v2) [[Pytorch]](https://github.com/MyeongJin-Kim/Learning-Texture-Invariant-Representation)
- xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation [[CVPR2020]](https://arxiv.org/abs/1911.12676) [[Demo]](https://www.youtube.com/watch?v=WgvBBCEKQVE) [[code]](https://github.com/valeoai/xmuda)
- Unsupervised Scene Adaptation with Memory Regularization in vivo [[IJCAI2020]](https://arxiv.org/abs/1912.11164) [[code]](https://github.com/layumi/Seg-Uncertainty)
- Joint Adversarial Learning for Domain Adaptation in Semantic Segmentation [[AAAI2020]](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-ZhangY.4858.pdf)
- An Adversarial Perturbation Oriented Domain Adaptation Approach for Semantic Segmentation [[AAAI2020]](https://arxiv.org/abs/1912.08954v1)
- Category Anchor-Guided Unsupervised Domain Adaptation for Semantic Segmentation [[NeurIPS2019]](https://arxiv.org/abs/1910.13049) [[code]](https://github.com/RogerZhangzz/CAG_UDA)
- MLSL: Multi-Level Self-Supervised Learning for Domain Adaptation with Spatially Independent and Semantically Consistent Labeling [[WACV2020]](https://arxiv.org/abs/1909.13776)
- Domain Bridge for Unpaired Image-to-Image Translation and Unsupervised Domain Adaptation [[WACV2020]](https://arxiv.org/abs/1910.10563)
- Guided Curriculum Model Adaptation and Uncertainty-Aware Evaluation for
Semantic Nighttime Image Segmentation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Sakaridis_Guided_Curriculum_Model_Adaptation_and_Uncertainty-Aware_Evaluation_for_Semantic_Nighttime_ICCV_2019_paper.pdf)
- Constructing Self-motivated Pyramid Curriculums for Cross-Domain Semantic
Segmentation: A Non-Adversarial Approach [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lian_Constructing_Self-Motivated_Pyramid_Curriculums_for_Cross-Domain_Semantic_Segmentation_A_Non-Adversarial_ICCV_2019_paper.pdf) [[Pytorch]](https://github.com/lianqing11/pycda)
- SSF-DAN: Separated Semantic Feature Based Domain Adaptation Network for Semantic Segmentation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Du_SSF-DAN_Separated_Semantic_Feature_Based_Domain_Adaptation_Network_for_Semantic_ICCV_2019_paper.pdf)
- Significance-aware Information Bottleneck for Domain Adaptive Semantic Segmentation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Luo_Significance-Aware_Information_Bottleneck_for_Domain_Adaptive_Semantic_Segmentation_ICCV_2019_paper.pdf)
- Domain Adaptation for Semantic Segmentation with Maximum Squares Loss [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Domain_Adaptation_for_Semantic_Segmentation_With_Maximum_Squares_Loss_ICCV_2019_paper.pdf) [[Pytorch]](https://github.com/ZJULearning/MaxSquareLoss)
- Self-Ensembling with GAN-based Data Augmentation for Domain Adaptation in Semantic Segmentation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Self-Ensembling_With_GAN-Based_Data_Augmentation_for_Domain_Adaptation_in_Semantic_ICCV_2019_paper.pdf)
- DADA: Depth-aware Domain Adaptation in Semantic Segmentation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Vu_DADA_Depth-Aware_Domain_Adaptation_in_Semantic_Segmentation_ICCV_2019_paper.pdf) [[code]](https://github.com/valeoai/DADA)
- Domain Adaptation for Structured Output via Discriminative Patch Representations [[ICCV2019 Oral]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Tsai_Domain_Adaptation_for_Structured_Output_via_Discriminative_Patch_Representations_ICCV_2019_paper.pdf) [[Project]](https://sites.google.com/site/yihsuantsai/research/iccv19-adapt-seg)
- Not All Areas Are Equal: Transfer Learning for Semantic Segmentation via Hierarchical Region Selection [[CVPR2019(Oral)]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Not_All_Areas_Are_Equal_Transfer_Learning_for_Semantic_Segmentation_CVPR_2019_paper.pdf)
- CrDoCo: Pixel-level Domain Transfer with Cross-Domain Consistency [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_CrDoCo_Pixel-Level_Domain_Transfer_With_Cross-Domain_Consistency_CVPR_2019_paper.pdf) [[Project]](https://yunchunchen.github.io/CrDoCo/)
- Bidirectional Learning for Domain Adaptation of Semantic Segmentation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Bidirectional_Learning_for_Domain_Adaptation_of_Semantic_Segmentation_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/liyunsheng13/BDL)
- Learning Semantic Segmentation from Synthetic Data: A Geometrically Guided Input-Output Adaptation Approach [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Semantic_Segmentation_From_Synthetic_Data_A_Geometrically_Guided_Input-Output_CVPR_2019_paper.pdf)
- All about Structure: Adapting Structural Information across Domains for Boosting Semantic Segmentation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_All_About_Structure_Adapting_Structural_Information_Across_Domains_for_Boosting_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/a514514772/DISE-Domain-Invariant-Structure-Extraction)
- DLOW: Domain Flow for Adaptation and Generalization [[CVPR2019 Oral]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gong_DLOW_Domain_Flow_for_Adaptation_and_Generalization_CVPR_2019_paper.pdf)
- Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation [[CVPR2019 Oral]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Luo_Taking_a_Closer_Look_at_Domain_Shift_Category-Level_Adversaries_for_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/RoyalVane/CLAN)
- ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation [[CVPR2019 Oral]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Vu_ADVENT_Adversarial_Entropy_Minimization_for_Domain_Adaptation_in_Semantic_Segmentation_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/valeoai/ADVENT)
- SPIGAN: Privileged Adversarial Learning from Simulation [[ICLR2019]](https://openreview.net/forum?id=rkxoNnC5FQ)
- Penalizing Top Performers: Conservative Loss for Semantic Segmentation Adaptation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xinge_Zhu_Penalizing_Top_Performers_ECCV_2018_paper.pdf)
- Domain transfer through deep activation matching [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Haoshuo_Huang_Domain_transfer_through_ECCV_2018_paper.pdf)
- Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yang_Zou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.pdf) [[Pytorch]](https://github.com/yzou2/CBST)
- DCAN: Dual channel-wise alignment networks for unsupervised scene adaptation [[ECCV2018]](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Zuxuan_Wu_DCAN_Dual_Channel-wise_ECCV_2018_paper.pdf) 
- Fully convolutional adaptation networks for semantic
segmentation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Fully_Convolutional_Adaptation_CVPR_2018_paper.pdf)
- Learning to Adapt Structured Output Space for Semantic Segmentation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tsai_Learning_to_Adapt_CVPR_2018_paper.pdf) [[Pytorch]](https://github.com/wasidennis/AdaptSegNet)
- Conditional Generative Adversarial Network for Structured Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hong_Conditional_Generative_Adversarial_CVPR_2018_paper.pdf)
- Learning From Synthetic Data: Addressing Domain Shift for Semantic Segmentation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sankaranarayanan_Learning_From_Synthetic_CVPR_2018_paper.pdf) [[Pytorch]](https://github.com/swamiviv/LSD-seg)
- Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes [[ICCV2017]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Curriculum_Domain_Adaptation_ICCV_2017_paper.pdf) [[Journal Version]](https://arxiv.org/abs/1812.09953v3) [[Keras]](https://github.com/YangZhang4065/AdaptationSeg)
- No more discrimination: Cross city adaptation of road scene segmenters [[ICCV2017]](http://openaccess.thecvf.com/content_ICCV_2017/supplemental/Chen_No_More_Discrimination_ICCV_2017_supplemental.pdf)

**Journal**
- Semantic-aware short path adversarial training for cross-domain semantic segmentation [[Neurocomputing 2019]](https://www.sciencedirect.com/science/article/pii/S0925231219315656#fig0002) 
- Weakly Supervised Adversarial Domain Adaptation for Semantic Segmentation in Urban Scenes [[TIP]](https://arxiv.org/abs/1904.09092v1)

**Arxiv**
- Rectifying Pseudo Label Learning via Uncertainty Estimation for Domain Adaptive Semantic Segmentation 
[[8 Mar 2020]](https://arxiv.org/abs/2003.03773)
- Class-Conditional Domain Adaptation on Semantic Segmentation [[27 Nov 2019]](https://arxiv.org/abs/1911.11981v1)
- Adversarial Learning and Self-Teaching Techniques for Domain Adaptation in Semantic Segmentation [[2 Sep 2019]](https://arxiv.org/abs/1909.00781v1)
- FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation [[8 Dec 2016]](https://arxiv.org/abs/1612.02649)

### Person Re-identification
**Arxiv**
- Structured Domain Adaptation for Unsupervised Person Re-identification [[arXiv 14 Mar 2020]](https://arxiv.org/abs/2003.06650)
- Domain Adaptive Attention Model for Unsupervised Cross-Domain Person Re-Identification [[arXiv 25 May 2019]](https://arxiv.org/abs/1905.10529)
- Camera Adversarial Transfer for Unsupervised Person Re-Identification [[arXiv 2 Apr 2019]](https://arxiv.org/abs/1904.01308)
- EANet: Enhancing Alignment for Cross-Domain Person Re-identification [[arXiv 29 Dec 2018]](https://arxiv.org/abs/1812.11369) [[Pytorch]](https://github.com/huanghoujing/EANet)
- One Shot Domain Adaptation for Person Re-Identification [[arXiv 26 Nov 2018]](https://arxiv.org/abs/1811.10144v1)
- Similarity-preserving Image-image Domain Adaptation for Person Re-identification [[arXiv 26 Nov 2018]](https://arxiv.org/abs/1811.10551v1)

**Conference**
- Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification [[ICLR2020]](https://openreview.net/forum?id=rJlnOhVYPS) [[Pytorch]](https://github.com/yxgeee/MMT)
- Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification [[ICCV2019 Oral]](https://arxiv.org/abs/1811.10144) [[Pytorch]](https://github.com/OasisYang/SSG)
- A Novel Unsupervised Camera-aware Domain Adaptation Framework for Person Re-identification [[ICCV2019]](https://arxiv.org/abs/1904.03425)
- Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification [[CVPR2019]](https://arxiv.org/abs/1904.01990v1) [[Pytorch]](https://github.com/zhunzhong07/ECN)
- Domain Adaptation through Synthesis for Unsupervised Person Re-identification [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Slawomir_Bak_Domain_Adaptation_through_ECCV_2018_paper.pdf)
- Person Transfer GAN to Bridge Domain Gap for Person Re-Identification [[CVPR2018]](https://arxiv.org/abs/1711.08565v2) 
- Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification [[CVPR2018]](https://arxiv.org/abs/1711.07027v3)

### Video Domain Adaptation

**Arxiv**
- Image to Video Domain Adaptation Using Web Supervision [[5 Aug 2019]](https://arxiv.org/abs/1908.01449)

**Conference**
- Adversarial Cross-Domain Action Recognition with Co-Attention [[AAAI2020]](https://arxiv.org/abs/1912.10405v1)
- Temporal Attentive Alignment for Large-Scale Video Domain Adaptation [[ICCV2019 Oral]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Temporal_Attentive_Alignment_for_Large-Scale_Video_Domain_Adaptation_ICCV_2019_paper.pdf) [[Pytorch]](https://github.com/olivesgatech/TA3N)
- Temporal Attentive Alignment for Video Domain Adaptation [[CVPRW 2019]](https://arxiv.org/abs/1905.10861v5) [[Pytorch]](https://github.com/olivesgatech/TA3N)

### Medical Related
**Arxiv**
- Unsupervised Domain Adaptation via Disentangled Representations: Application to Cross-Modality Liver Segmentation [[arXiv 29 Aug 2019]](https://arxiv.org/abs/1907.13590)
- Synergistic Image and Feature Adaptation: Towards Cross-Modality Domain Adaptation for Medical Image Segmentation [[arXiv on 24 Jan 2019]](https://arxiv.org/abs/1901.08211v1)
- Unsupervised domain adaptation for medical imaging segmentation with self-ensembling [[arXiv 14 Nov 2018]](https://arxiv.org/abs/1811.06042v1)

**Conference**
- Semantic-Transferable Weakly-Supervised Endoscopic Lesions Segmentation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Dong_Semantic-Transferable_Weakly-Supervised_Endoscopic_Lesions_Segmentation_ICCV_2019_paper.pdf)

### Monocular Depth Estimation
- Geometry-Aware Symmetric Domain Adaptation for Monocular Depth Estimation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Geometry-Aware_Symmetric_Domain_Adaptation_for_Monocular_Depth_Estimation_CVPR_2019_paper.pdf)
- Real-Time Monocular Depth Estimation using Synthetic Data with Domain Adaptation via Image Style Transfer [[CVPR2018]](http://breckon.eu/toby/publications/papers/abarghouei18monocular.pdf)

### 3D Reconstruction
**Conference**
- Domain-Adaptive Single-View 3D Reconstruction [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Pinheiro_Domain-Adaptive_Single-View_3D_Reconstruction_ICCV_2019_paper.pdf)

### Others
**Arxiv**
- Open-Ended Visual Question Answering by Multi-Modal Domain Adaptation [[11 Nov 2019]](https://arxiv.org/abs/1911.04058)
- DANE: Domain Adaptive Network Embedding [[arXiv 3 Jun 2019]](https://arxiv.org/abs/1906.00684v1)
- Active Adversarial Domain Adaptation [[arXiv 16 Apr 2019]](https://arxiv.org/abs/1904.07848v1)

**Conference**
- Unsupervised Domain Adaptation for 3D Human Pose Estimation [[ACM MM2019]](https://dl.acm.org/citation.cfm?id=3351052)
- PointDAN: A Multi-Scale 3D Domain Adaption Network for Point Cloud Representation [[NeurIPS 2019]](https://arxiv.org/abs/1911.02744v1) [[code]](https://github.com/canqin001/PointDAN)
- Deep Head Pose Estimation Using Synthetic Images and Partial Adversarial Domain Adaption for Continuous Label Spaces [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Kuhnke_Deep_Head_Pose_Estimation_Using_Synthetic_Images_and_Partial_Adversarial_ICCV_2019_paper.pdf)
- Cross-Domain Adaptation for Animal Pose Estimation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cao_Cross-Domain_Adaptation_for_Animal_Pose_Estimation_ICCV_2019_paper.pdf)
- GA-DAN: Geometry-Aware Domain Adaptation Network for Scene Text Detection and Recognition [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhan_GA-DAN_Geometry-Aware_Domain_Adaptation_Network_for_Scene_Text_Detection_and_ICCV_2019_paper.pdf)
- Accelerating Deep Unsupervised Domain Adaptation with Transfer Channel Pruning [[IJCNN]](https://arxiv.org/abs/1904.02654)
- Adversarial Adaptation of Scene Graph Models for Understanding Civic Issues [[WWW2019]](https://arxiv.org/abs/1901.10124)
- Cross-Dataset Adaptation for Visual Question Answering [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chao_Cross-Dataset_Adaptation_for_CVPR_2018_paper.pdf)

## Related Topics
### Image-to-Image Translation
**Arxiv**
- MISO: Mutual Information Loss with Stochastic Style Representations for Multimodal Image-to-Image Translation [[arXiv 11 Feb 2019]](https://arxiv.org/abs/1902.03938)
- TraVeLGAN: Image-to-image Translation by Transformation Vector Learning [[arXiv 25 Feb 2019]](https://arxiv.org/abs/1902.09631)

**Conference**
- Batch Weight for Domain Adaptation With Mass Shift [[ICCV2019]](https://arxiv.org/abs/1905.12760)
- Emerging Disentanglement in Auto-Encoder Based Unsupervised Image Content Transfer [[ICLR2019]](https://openreview.net/forum?id=BylE1205Fm) [[Pytorch]](https://github.com/oripress/ContentDisentanglement)
- Unsupervised Attention-guided Image-to-Image Translation [[NIPS2018]](https://papers.nips.cc/paper/7627-unsupervised-attention-guided-image-to-image-translation)
- Image-to-image translation for cross-domain disentanglement [[NIPS2018]](https://papers.nips.cc/paper/7404-image-to-image-translation-for-cross-domain-disentanglement)
- One-Shot Unsupervised Cross Domain Translation [[NIPS2018]](http://papers.nips.cc/paper/7480-one-shot-unsupervised-cross-domain-translation)
- A Unified Feature Disentangler for Multi-Domain Image Translation and Manipulation [[NIPS2018]](http://papers.nips.cc/paper/7525-a-unified-feature-disentangler-for-multi-domain-image-translation-and-manipulation)
- Unsupervised Image-to-Image Translation Using Domain-Specific Variational Information Bound [[NIPS2018]](http://papers.nips.cc/paper/8236-unsupervised-image-to-image-translation-using-domain-specific-variational-information-bound)
- Multi-view Adversarially Learned Inference for Cross-domain Joint Distribution Matching [[KDD2018]](http://www.kdd.org/kdd2018/accepted-papers/view/multi-view-adversarially-learned-inference-for-cross-domain-joint-distribut)
- Unpaired Multi-Domain Image Generation via Regularized Conditional GANs [[IJCAI2018]](https://www.ijcai.org/proceedings/2018/0354.pdf) [[TensorFlow]](https://github.com/xudonmao/RegCGAN)
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

### Disentangled Representation Learning
**Arxiv**
- Towards a Definition of Disentangled Representations [[arXiv 5 Dec 2018]](https://arxiv.org/abs/1812.02230)

**Conference**
- Emerging Disentanglement in Auto-Encoder Based Unsupervised Image Content Transfer [[ICLR2019]](https://openreview.net/forum?id=BylE1205Fm) [[Pytorch]](https://github.com/oripress/ContentDisentanglement)
- Life-Long Disentangled Representation Learning with Cross-Domain Latent Homologies [[NIPS2018]](https://papers.nips.cc/paper/8193-life-long-disentangled-representation-learning-with-cross-domain-latent-homologies)
- Image-to-image translation for cross-domain disentanglement [[NIPS2018]](https://papers.nips.cc/paper/7404-image-to-image-translation-for-cross-domain-disentanglement)


## Benchmarks
- Syn2Real: A New Benchmark forSynthetic-to-Real Visual Domain Adaptation [[arXiv 26 Jun]](https://arxiv.org/abs/1806.09755v1) [[Project]](http://ai.bu.edu/syn2real/)

# Library
- [Xlearn: Transfer Learning Library](https://github.com/thuml/Xlearn)
- [deep-transfer-learning: a PyTorch library for deep transfer learning](https://github.com/easezyc/deep-transfer-learning)
- [salad: a Semi-supervised Adaptive Learning Across Domains](https://domainadaptation.org/)
- [Dassl: a PyTorch toolbox for domain adaptation and semi-supervised learning](https://github.com/KaiyangZhou/Dassl.pytorch)

# Lectures and Tutorials
- A Primer on Domain Adaptation [[PDF]](https://arxiv.org/abs/2001.09994v2)


# Other Resources
- [transferlearning](https://github.com/jindongwang/transferlearning)
