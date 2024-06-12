# awesome-domain-adaptation

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 

This repo is a collection of AWESOME things about domain adaptation, including papers, code, etc. Feel free to star and fork.

# Contents
- [awesome-domain-adaptation](#awesome-domain-adaptation)
- [Contents](#contents)
- [Papers](#papers)
  - [Survey](#survey)
  - [Theory](#theory)
  - [Explainable](#explainable)
  - [Unsupervised DA](#unsupervised-da)
    - [Adversarial Methods](#adversarial-methods)
    - [Distance-based Methods](#distance-based-methods)
    - [Information-based Methods](#information-based-methods)
    - [Optimal Transport](#optimal-transport)
    - [Incremental Methods](#incremental-methods)
    - [Semi-Supervised-Learning-Based Methods](#semi-supervised-learning-based-methods)
    - [Self-training-Based Methods](#self-training-based-methods)
    - [Self-Supervised Methods](#self-supervised-methods)
    - [Transformer-based Methods](#transformer-based-methods)
    - [Other Methods](#other-methods)
  - [Semi-supervised DA](#semi-supervised-da)
  - [Weakly-Supervised DA](#weakly-supervised-da)
  - [Zero-shot DA](#zero-shot-da)
  - [One-shot DA](#one-shot-da)
  - [Few-shot UDA](#few-shot-uda)
  - [Few-shot DA](#few-shot-da)
  - [Partial DA](#partial-da)
  - [Open Set DA](#open-set-da)
  - [Universal DA](#universal-da)
  - [Open Compound DA](#open-compound-da)
  - [Multi Source DA](#multi-source-da)
  - [Multi Target DA](#multi-target-da)
  - [Incremental DA](#incremental-da)
  - [Multi Step DA](#multi-step-da)
  - [Heterogeneous DA](#heterogeneous-da)
  - [Target-agnostic DA](#target-agnostic-da)
  - [Federated DA](#federated-da)
  - [Continuously Indexed DA](#continuously-indexed-da)
  - [Source Free DA](#source-free-da)
  - [Active DA](#active-da)
  - [Generalized Domain Adaptation](#generalized-domain-adaptation)
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
    - [Sim-to-Real Transfer](#sim-to-real-transfer)
    - [Video Domain Adaptation](#video-domain-adaptation)
    - [Medical Related](#medical-related)
    - [Monocular Depth Estimation](#monocular-depth-estimation)
    - [3D](#3d)
    - [Fine-Grained Domain](#fine-grained-domain)
    - [LiDAR](#lidar)
    - [Remote Sensing](#remote-sensing)
    - [Others](#others-1)
  - [Benchmarks](#benchmarks)
- [Library](#library)
- [Lectures and Tutorials](#lectures-and-tutorials)
- [Other Resources](#other-resources)

# Papers
## Survey
**Arxiv**
- Video Unsupervised Domain Adaptation with Deep Learning: A Comprehensive Survey [[17 Nov 2022]](https://arxiv.org/abs/2211.10412) [[project]](https://github.com/xuyu0010/awesome-video-domain-adaptation)
- A Survey on Deep Domain Adaptation for LiDAR Perception [[7 Jun 2021]](https://arxiv.org/abs/2106.02377)
- A Comprehensive Survey on Transfer Learning [[7 Nov 2019]](https://arxiv.org/abs/1911.02685)
- Transfer Adaptation Learning: A Decade Survey [[12 Mar 2019]](https://arxiv.org/abs/1903.04687)
- A review of single-source unsupervised domain adaptation [[16 Jan 2019]](https://arxiv.org/abs/1901.05335)
- An introduction to domain adaptation and transfer learning [[31 Dec 2018]](https://arxiv.org/abs/1812.11806v2)
- A Survey of Unsupervised Deep Domain Adaptation [[6 Dec 2018]](https://arxiv.org/abs/1812.02849v2)
- Transfer Learning for Cross-Dataset Recognition: A Survey [[2017]](https://sci-hub.tw/https://arxiv.org/abs/1705.04396)
- Domain Adaptation for Visual Applications: A Comprehensive Survey  [[2017]](https://arxiv.org/abs/1702.05374)

**Journal**
- Survey on Unsupervised Domain Adaptation for Semantic Segmentation for Visual Perception in Automated Driving [[IEEE Access 2023]](https://ieeexplore.ieee.org/document/10128983)
- A Review of Single-Source Deep Unsupervised Visual Domain Adaptation [[TNNLS 2020]](https://arxiv.org/pdf/2009.00155.pdf)
- Deep Visual Domain Adaptation: A Survey [[Neurocomputing 2018]](https://arxiv.org/abs/1802.03601v4)
- A Survey on Deep Transfer Learning [[ICANN2018]](https://arxiv.org/abs/1808.01974v1)
- Visual domain adaptation: A survey of recent advances [[2015]](https://sci-hub.tw/10.1109/msp.2014.2347059)

## Theory
**Arxiv**
- A Theory of Label Propagation for Subpopulation Shift [[22 Feb 2021]](https://arxiv.org/abs/2102.11203)
- A General Upper Bound for Unsupervised Domain Adaptation [[3 Oct 2019]](https://arxiv.org/abs/1910.01409)
- On Deep Domain Adaptation: Some Theoretical Understandings [[arXiv 15 Nov 2018]](https://arxiv.org/abs/1811.06199)

**Conference**
- Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift [[NeurIPS 2020]](https://arxiv.org/abs/2003.04475)
- Bridging Theory and Algorithm for Domain Adaptation [[ICML2019]](http://proceedings.mlr.press/v97/zhang19i/zhang19i.pdf) [[Pytorch]](https://github.com/thuml/MDD)
- On Learning Invariant Representation for Domain Adaptation [[ICML2019]](https://arxiv.org/abs/1901.09453v1) [[code]](https://github.com/KeiraZhao/On-Learning-Invariant-Representations-for-Domain-Adaptation)
- Unsupervised Domain Adaptation Based on Source-guided Discrepancy [[AAAI2019]](https://arxiv.org/abs/1809.03839)
- Learning Bounds for Domain Adaptation [[NIPS2007]](http://papers.nips.cc/paper/3212-learning-bounds-for-domain-adaptation)
- Analysis of Representations for Domain Adaptation [[NIPS2006]](https://papers.nips.cc/paper/2983-analysis-of-representations-for-domain-adaptation)

**Journal**
- On a Regularization of Unsupervised Domain Adaptation in RKHS [[ACHA2021]](https://www.sciencedirect.com/science/article/abs/pii/S1063520321001032?via%3Dihub)
- Unsupervised Multi-Class Domain Adaptation: Theory, Algorithms, and Practice [[TPAMI2020]](https://arxiv.org/abs/2002.08681) [[PyTroch]](https://github.com/YBZh/MultiClassDA)
- On generalization in moment-based domain adaptation [[AMAI2020]](https://link.springer.com/article/10.1007/s10472-020-09719-x)
- A theory of learning from different domains [[ML2010]](https://link.springer.com/content/pdf/10.1007%2Fs10994-009-5152-4.pdf)


## Explainable
**Conference**
- Visualizing Adapted Knowledge in Domain Transfer [[CVPR2021]](https://arxiv.org/abs/2104.10602) [[Pytorch]](https://github.com/hou-yz/DA_visualization)

## Unsupervised DA

### Adversarial Methods

**Conference**
- SPA: A Graph Spectral Alignment Perspective for Domain Adaptation [[NeurIPS 2023]](https://arxiv.org/abs/2310.17594) [[Pytorch]](https://github.com/CrownX/SPA)
- Reusing the Task-specific Classifier as a Discriminator: Discriminator-free Adversarial Domain Adaptation [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Reusing_the_Task-Specific_Classifier_as_a_Discriminator_Discriminator-Free_Adversarial_Domain_CVPR_2022_paper.pdf) [[Pytorch]](https://github.com/xiaoachen98/DALN)
- A Closer Look at Smoothness in Domain Adversarial Training [[ICML2022]](https://arxiv.org/abs/2206.08213) [[Pytorch]](https://github.com/val-iisc/SDAT)
- ToAlign: Task-oriented Alignment for Unsupervised Domain Adaptation [[NeurIPS2021]](https://arxiv.org/abs/2004.01888) [[Pytorch]](https://github.com/microsoft/UDA)
- Adversarial Unsupervised Domain Adaptation With Conditional and Label Shift: Infer, Align and Iterate [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Adversarial_Unsupervised_Domain_Adaptation_With_Conditional_and_Label_Shift_Infer_ICCV_2021_paper.html)
- Gradient Distribution Alignment Certificates Better Adversarial Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Gao_Gradient_Distribution_Alignment_Certificates_Better_Adversarial_Domain_Adaptation_ICCV_2021_paper.html)
- Re-energizing Domain Discriminator with Sample Relabeling
for Adversarial Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Jin_Re-Energizing_Domain_Discriminator_With_Sample_Relabeling_for_Adversarial_Domain_Adaptation_ICCV_2021_paper.pdf)
- Cross-Domain Gradient Discrepancy Minimization for Unsupervised Domain Adaptation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Du_Cross-Domain_Gradient_Discrepancy_Minimization_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.pdf) [[Pytorch]](https://github.com/lijin118/CGDM)
- MetaAlign: Coordinating Domain Alignment and Classification for Unsupervised Domain Adaptation [[CVPR2021]](https://arxiv.org/abs/2103.13575) [[Pytorch]](https://github.com/microsoft/UDA)
- Self-adaptive Re-weighted Adversarial Domain Adaptation [[IJCAI2020]](https://www.ijcai.org/Proceedings/2020/0440.pdf)
- DIRL: Domain-Invariant Reperesentation Learning Approach for Sim-to-Real Transfer [[CoRL2020]](https://arxiv.org/abs/2011.07589) [[Project]](https://www.sites.google.com/view/dirl)
- SSA-DA: Bi-dimensional feature alignment for cross-domain object detection [[ECCV Workshop 2020]](https://arxiv.org/pdf/2011.07205.pdf)
- Classes Matter: A Fine-grained Adversarial Approach to Cross-domain Semantic Segmentation [[ECCV2020]](https://arxiv.org/abs/2007.09222) [[PyTorch]](https://github.com/JDAI-CV/FADA)
- MCAR: Adaptive object detection with dual multi-label prediction [[ECCV2020]](https://arxiv.org/pdf/2003.12943.pdf)
- Gradually Vanishing Bridge for Adversarial Domain Adaptation [[CVPR2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cui_Gradually_Vanishing_Bridge_for_Adversarial_Domain_Adaptation_CVPR_2020_paper.pdf) [[Pytorch]](https://github.com/cuishuhao/GVB)
- Implicit Class-Conditioned Domain Alignment for Unsupervised Domain Adaptation [[ICML2020]](https://arxiv.org/abs/2006.04996) [[Pytorch]](https://github.com/xiangdal/implicit_alignment)
- Adversarial-Learned Loss for Domain Adaptation [[AAAI2020]](https://arxiv.org/abs/2001.01046v1)
- Structure-Aware Feature Fusion for Unsupervised Domain Adaptation [[AAAI2020]](https://aaai.org/Papers/AAAI/2020GB/AAAI-ChenQ.8923.pdf)
- Adversarial Domain Adaptation with Domain Mixup [[AAAI2020]](https://arxiv.org/abs/1912.01805v1) [[Pytorch]](https://github.com/ChrisAllenMing/Mixup_for_UDA)
- Discriminative Adversarial Domain Adaptation [[AAAI2020]](https://arxiv.org/abs/1911.12036v1) [[Pytorch]](https://github.com/huitangtang/DADA-AAAI2020)
- Bi-Directional Generation for Unsupervised Domain Adaptation [[AAAI2020]](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-YangG.1084.pdf)
- Cross-stained Segmentation from Renal Biopsy Images Using Multi-level Adversarial Learning [[ICASSP 2020]](https://arxiv.org/abs/2002.08587)
- Curriculum based Dropout Discriminator for Domain Adaptation [[BMVC2019]](https://arxiv.org/pdf/1907.10628.pdf) [[Project]](https://delta-lab-iitk.github.io/CD3A/)
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
- Incremental Unsupervised Domain-Adversarial Training of Neural Networks [[TNNLS 2020]](https://ieeexplore.ieee.org/document/9216604)
- Unsupervised Multi-Class Domain Adaptation: Theory, Algorithms, and Practice [[TPAMI2020]](https://arxiv.org/abs/2002.08681) [[PyTroch]](https://github.com/YBZh/MultiClassDA)
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
- Robust unsupervised domain adaptation for neural networks via moment alignment [[InfSc2019]](https://www.sciencedirect.com/science/article/abs/pii/S0020025519300301)

**Conference**
- Domain Conditioned Adaptation Network [[AAAI2020]](https://arxiv.org/abs/2005.06717) [[Pytorch]](https://github.com/BIT-DA/DCAN)
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


### Information-based Methods
- Hypothesis Disparity Regularized Mutual Information Maximization [[AAAI2021]](https://arxiv.org/abs/2012.08072)


### Optimal Transport
**Conference**
- Global-Local Regularization Via Distributional Robustness [[AISTATS2023]](https://arxiv.org/abs/2203.00553) [[Pytorch]](https://github.com/VietHoang1512/GLOT/)
- MOST: Multi-Source Domain Adaptation via Optimal Transport for Student-Teacher Learning [[UAI2021]](https://auai.org/uai2021/pdf/uai2021.106.pdf)
- LAMDA: Label Matching Deep Domain Adaptation [[ICML2021]](http://proceedings.mlr.press/v139/le21a.html)
- TIDOT: A Teacher Imitation Learning Approach for Domain Adaptation with Optimal Transport [[IJCAI2021]](https://www.ijcai.org/proceedings/2021/0394.pdf) 
- Unbalanced minibatch Optimal Transport; applications to Domain Adaptation [[ICML2021]](https://arxiv.org/abs/2103.03606) [[Pytorch]](https://github.com/kilianFatras/JUMBOT)
- Graph Optimal Transport for Cross-Domain Alignment [[ICML2020]](https://proceedings.icml.cc/static/paper_files/icml/2020/971-Paper.pdf)
- Margin-aware Adversarial Domain Adaptation with Optimal Transport [[ICML2020]](https://proceedings.icml.cc/static/paper_files/icml/2020/2666-Paper.pdf) [[code]](https://github.com/sofiendhouib/MADAOT)
- Metric Learning in Optimal Transport for Domain Adaptation [[IJCAI2020]](https://www.ijcai.org/Proceedings/2020/0299.pdf)
- Reliable Weighted Optimal Transport for Unsupervised Domain Adaptation [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Reliable_Weighted_Optimal_Transport_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf)
- Enhanced Transport Distance for Unsupervised Domain Adaptation [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Enhanced_Transport_Distance_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf) [[Pytorch]](https://github.com/yimzhai3/ETD)
- Differentially Private Optimal Transport: Application to Domain Adaptation [[IJCAI2019]](https://www.ijcai.org/proceedings/2019/0395.pdf)
- DeepJDOT: Deep Joint distribution optimal transport for unsupervised domain adaptation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bharath_Bhushan_Damodaran_DeepJDOT_Deep_Joint_ECCV_2018_paper.pdf) [[Keras]](https://github.com/bbdamodaran/deepJDOT)
- Joint Distribution Optimal Transportation for Domain Adaptation [[NIPS2017]](http://papers.nips.cc/paper/6963-joint-distribution-optimal-transportation-for-domain-adaptation.pdf) [[python]](https://github.com/rflamary/JDOT) [[Python Optimal Transport Library]](https://github.com/rflamary/POT)

**Arxiv**
- CDOT: Continuous Domain Adaptation using Optimal Transport [[20 Sep 2019]](https://arxiv.org/abs/1909.11448)


### Incremental Methods
- Incremental Unsupervised Domain-Adversarial Training of Neural Networks [[TNNLS 2020]](https://ieeexplore.ieee.org/document/9216604)


### Semi-Supervised-Learning-Based Methods
- Label Propagation with Augmented Anchors: A Simple Semi-Supervised Learning baseline for Unsupervised Domain Adaptation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490749.pdf)
- Semi-supervised Models are Strong Unsupervised Domain Adaptation Learners [[arXiv 2021]]((https://arxiv.org/pdf/2106.00417.pdf))[[Pytorch]](https://github.com/YBZh/Bridging_UDA_SSL)


### Self-training-Based Methods
- Cycle Self-Training for Domain Adaptation [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/c1fea270c48e8079d8ddf7d06d26ab52-Abstract.html)
- Meta Self-Learning for Multi-Source Domain Adaptation: A Benchmark [[ICCV Workshop 2021]](https://arxiv.org/abs/2108.10840) [[Pytorch]](https://github.com/bupt-ai-cz/Meta-SelfLearning)
- Instance Adaptive Self-Training for Unsupervised Domain Adaptation [[ECCV 2020]](https://arxiv.org/abs/2008.12197) [[Pytorch]](https://github.com/bupt-ai-cz/IAST-ECCV2020)
- Self-training Avoids Using Spurious Features Under Domain Shift [[NeurIPS 2020]](https://arxiv.org/abs/2006.10032)
- Two-phase Pseudo Label Densification for Self-training based Domain Adaptation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580528.pdf)

**Arxiv**
- Probabilistic Contrastive Learning for Domain Adaptation [[arXiv 20211]](https://arxiv.org/abs/2111.06021) [[Pytorch]](https://github.com/ljjcoder/Probabilistic-Contrastive-Learning)
- Gradual Domain Adaptation via Self-Training of Auxiliary Models[[arXiv 2021]](https://arxiv.org/pdf/2106.09890.pdf)[[Pytorch]](https://github.com/YBZh/AuxSelfTrain)


### Self-Supervised Methods
**Conference**
- Self-Supervised CycleGAN for Object-Preserving Image-to-Image Domain Adaptation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650494.pdf)

**Arxiv**
- Unsupervised Domain Adaptation through Self-Supervision [[arXiv 26 Sep 2019]](https://arxiv.org/abs/1909.11825)

### Transformer-based Methods
**Conference**
- Safe Self-Refinement for Transformer-Based Domain Adaptation [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Safe_Self-Refinement_for_Transformer-Based_Domain_Adaptation_CVPR_2022_paper.pdf) [[Pytorch](https://github.com/tsun/SSRT)

### Other Methods
**Conference**
- Prior Knowledge Guided Unsupervised Domain Adaptation [[ECCV2022]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930628.pdf) [[Pytorch]](https://github.com/tsun/KUDA)
- Revisiting Unsupervised Domain Adaptation Models: a Smoothness Perspective [[ACCV2022]](https://openaccess.thecvf.com/content/ACCV2022/html/Wang_Revisiting_Unsupervised_Domain_Adaptation_Models_a_Smoothness_Perspective_ACCV_2022_paper.html) [[Pytorch]](https://github.com/Wang-Xiaodong1899/LeCo_UDA)
- Reducing the Covariate Shift by Mirror Samples in Cross Domain Alignment [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/4f284803bd0966cc24fa8683a34afc6e-Abstract.html)
- Pareto Domain Adaptation [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/6ba3af5d7b2790e73f0de32e5c8c1798-Abstract.html)
- ToAlign: Task-Oriented Alignment for Unsupervised Domain Adaptation [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/731c83db8d2ff01bdc000083fd3c3740-Abstract.html)
- A Prototype-Oriented Framework for Unsupervised Domain Adaptation [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/8edd72158ccd2a879f79cb2538568fdc-Abstract.html)
- Understanding the Limits of Unsupervised Domain Adaptation via Data Poisoning [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/90cc440b1b8caa520c562ac4e4bbcb51-Abstract.html)
- SENTRY: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Prabhu_SENTRY_Selective_Entropy_Optimization_via_Committee_Consistency_for_Unsupervised_Domain_ICCV_2021_paper.html)
- Transporting Causal Mechanisms for Unsupervised Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Yue_Transporting_Causal_Mechanisms_for_Unsupervised_Domain_Adaptation_ICCV_2021_paper.html)
- Semantic Concentration for Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Li_Semantic_Concentration_for_Domain_Adaptation_ICCV_2021_paper.html) 
- FixBi: Bridging Domain Spaces for Unsupervised Domain Adaptation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Na_FixBi_Bridging_Domain_Spaces_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.pdf)
- Domain Adaptation With Auxiliary Target Domain-Oriented Classifier [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Liang_Domain_Adaptation_With_Auxiliary_Target_Domain-Oriented_Classifier_CVPR_2021_paper.pdf)
- Conditional Bures Metric for Domain Adaptation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_Conditional_Bures_Metric_for_Domain_Adaptation_CVPR_2021_paper.pdf)
- DRANet: Disentangling Representation and Adaptation Networks for Unsupervised Cross-Domain Adaptation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_DRANet_Disentangling_Representation_and_Adaptation_Networks_for_Unsupervised_Cross-Domain_Adaptation_CVPR_2021_paper.pdf)
- Visualizing Adapted Knowledge in Domain Transfer [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Hou_Visualizing_Adapted_Knowledge_in_Domain_Transfer_CVPR_2021_paper.pdf) [[Pytorch]](https://github.com/hou-yz/DA_visualization)
- Instance Level Affinity-Based Transfer for Unsupervised Domain Adaptation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Sharma_Instance_Level_Affinity-Based_Transfer_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.pdf) [[code coming soon]](https://github.com/astuti/ILA-DA)
- Dynamic Domain Adaptation for Efficient Inference [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Dynamic_Domain_Adaptation_for_Efficient_Inference_CVPR_2021_paper.pdf) [[Pytorch]](https://github.com/BIT-DA/DDA)
- Transferable Semantic Augmentation for Domain Adaptation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Transferable_Semantic_Augmentation_for_Domain_Adaptation_CVPR_2021_paper.pdf) [[Pytorch]](https://github.com/BIT-DA/TSA)
- MetaAlign: Coordinating Domain Alignment and Classification for Unsupervised Domain Adaptation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wei_MetaAlign_Coordinating_Domain_Alignment_and_Classification_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.pdf)
- DRANet: Disentangling Representation and Adaptation Networks for Unsupervised Cross-Domain Adaptation [[CVPR2021]](https://arxiv.org/abs/2103.13447v1)
- Dynamic Weighted Learning for Unsupervised Domain Adaptation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Xiao_Dynamic_Weighted_Learning_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.pdf)
- Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift [[NeurIPS 2020]](https://arxiv.org/abs/2003.04475)
- Transferable Calibration with Lower Bias and Variance in Domain Adaptation [[NeurIPS 2020]](https://arxiv.org/abs/2007.08259)
- A Dictionary Approach to Domain-Invariant Learning in Deep Networks [[NeurIPS 2020]](https://arxiv.org/abs/1909.11285)
- Heuristic Domain Adaptation [[NeurIPS2020]](https://arxiv.org/abs/2011.14540) [[Pytorch]](https://github.com/cuishuhao/HDA)
- Unsupervised Domain Adaptation for Semantic Segmentation of NIR Images through Generative Latent Search [[ECCV2020]](https://arxiv.org/abs/2006.08696)[[code]](https://github.com/ambekarsameer96/GLSS)
- Mind the Discriminability: Asymmetric Adversarial Domain Adaptation [[ECCV2020]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690579.pdf)
- Domain2Vec: Domain Embedding for Unsupervised Domain Adaptation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510749.pdf)
- CSCL: Critical Semantic-Consistent Learning for Unsupervised Domain Adaptation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530732.pdf)
- Minimum Class Confusion for Versatile Domain Adaptation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660460.pdf)
- Partially-Shared Variational Auto-encoders for Unsupervised Domain Adaptation with Target Shift [[ECCV2020]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2472_ECCV_2020_paper.php) [[Pytorch]](https://github.com/iiyama-lab/PS-VAEs)
- Label Propagation with Augmented Anchors: A Simple Semi-Supervised Learning baseline for Unsupervised Domain Adaptation [[ECCV2020]](https://arxiv.org/pdf/2007.07695.pdf) [[PyTorch]](https://github.com/YBZh/Label-Propagation-with-Augmented-Anchors)
- Unsupervised Domain Adaptation via Structurally Regularized Deep Clustering [[CVPR2020 Oral]](http://arxiv.org/abs/2003.08607) [[Pytorch]](https://github.com/huitangtang/SRDC-CVPR2020)
- Towards Discriminability and Diversity: Batch Nuclear-norm Maximization under Label Insufficient Situations [[CVPR2020 Oral]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cui_Towards_Discriminability_and_Diversity_Batch_Nuclear-Norm_Maximization_Under_Label_Insufficient_CVPR_2020_paper.pdf) [[Pytorch]](https://github.com/cuishuhao/BNM)
- Unsupervised Domain Adaptation With Hierarchical Gradient Synchronization [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_Unsupervised_Domain_Adaptation_With_Hierarchical_Gradient_Synchronization_CVPR_2020_paper.pdf)
- Spherical Space Domain Adaptation With Robust Pseudo-Label Loss [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Gu_Spherical_Space_Domain_Adaptation_With_Robust_Pseudo-Label_Loss_CVPR_2020_paper.pdf) [[Pytorch]](https://github.com/XJTU-XGU/RSDA)
- Stochastic Classifiers for Unsupervised Domain Adaptation [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Lu_Stochastic_Classifiers_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf)
- Structure Preserving Generative Cross-Domain Learning [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Xia_Structure_Preserving_Generative_Cross-Domain_Learning_CVPR_2020_paper.pdf)
- Light-weight Calibrator: A Separable Component for Unsupervised Domain Adaptation [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Light-weight_Calibrator_A_Separable_Component_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf) [[code]](https://github.com/yeshaokai/Calibrator-Domain-Adaptation)
- Domain Adaptive Multiflow Networks [[ICLR2020]](https://openreview.net/forum?id=rJxycxHKDS)
- Unsupervised Domain Adaptation via Discriminative Manifold Embedding and Alignment [[AAAI2020]](https://arxiv.org/abs/2002.08675v1)
- Visual Domain Adaptation by Consensus-based Transfer to Intermediate Domain [[Paper]](https://aaai.org/Papers/AAAI/2020GB/AAAI-ChoiJ.3612.pdf)
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
- Domain Specific Batch Normalization for Unsupervised Domain Adaptation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_Domain-Specific_Batch_Normalization_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/wgchang/DSBN)
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
- Target-Independent Domain Adaptation for WBC Classification using Generative Latent Search [[IEEE TMI 2020]](https://ieeexplore.ieee.org/document/9139471)[[code]](https://github.com/prinshul/WBC-Classification-UDA)
- Adaptive Batch Normalization for practical domain adaptation [[Pattern Recognition(2018)]](https://www.sciencedirect.com/science/article/pii/S003132031830092X)
- Unsupervised Domain Adaptation by Mapped Correlation Alignment [[IEEE ACCESS]](https://ieeexplore.ieee.org/abstract/document/8434290/)

**Arxiv**
- Low-confidence Samples Matter for Domain Adaptation [[6 Feb 2022]](https://arxiv.org/abs/2202.02802) [[Pytorch]](https://github.com/zhyx12/MixLRCo)
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

## Foundation-Models based DA

**Conference**
-  POUF: Prompt-oriented unsupervised fine-tuning for large pre-trained models [[ICML2023]](https://arxiv.org/abs/2305.00350) [[Pytorch]](https://github.com/korawat-tanwisuth/POUF)

## Semi-supervised DA

**Conference**
- Semi-Supervised Domain Adaptation With Source Label Adaptation [[CVPR 2023]](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Semi-Supervised_Domain_Adaptation_With_Source_Label_Adaptation_CVPR_2023_paper.html)
- Multi-level Consistency Learning for Semi-supervised Domain Adaptation [[IJCAI 2022]](https://arxiv.org/abs/2205.04066)
- AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation [[ICLR 2022]](https://openreview.net/pdf?id=Q5uh1Nvv5dm)
- CLDA: Contrastive Learning for Semi-Supervised Domain Adaptation [[NeurIPS]](https://proceedings.neurips.cc/paper/2021/hash/288cd2567953f06e460a33951f55daaf-Abstract.html)
- Deep Co-Training With Task Decomposition for Semi-Supervised Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_Deep_Co-Training_With_Task_Decomposition_for_Semi-Supervised_Domain_Adaptation_ICCV_2021_paper.html)
- ECACL: A Holistic Framework for Semi-Supervised Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Li_ECACL_A_Holistic_Framework_for_Semi-Supervised_Domain_Adaptation_ICCV_2021_paper.html)
- Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Cross-Domain_Adaptive_Clustering_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.pdf)
- Semi-supervised Domain Adaptation based on Dual-level Domain Mixing for Semantic Segmentation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Semi-Supervised_Domain_Adaptation_Based_on_Dual-Level_Domain_Mixing_for_Semantic_CVPR_2021_paper.pdf)
- Learning Invariant Representations and Risks for Semi-supervised Domain Adaptation [[CVPR2021]](https://arxiv.org/abs/2010.04647)
- Improving Semi-Supervised Domain Adaptation Using Effective Target Selection and Semantics [[CVPRW2021]](https://openaccess.thecvf.com/content/CVPR2021W/LLID/papers/Singh_Improving_Semi-Supervised_Domain_Adaptation_Using_Effective_Target_Selection_and_Semantics_CVPRW_2021_paper.pdf) [[Code]](https://github.com/Anurag14/STar-framework)
- Attract, Perturb, and Explore: Learning a Feature Alignment Network for Semi-supervised Domain Adaptation [[ECCV2020]](https://arxiv.org/abs/2007.09375v1)
- Online Meta-Learning for Multi-Source and Semi-Supervised Domain Adaptation [[ECCV2020]](https://arxiv.org/abs/2004.04398)
- Bidirectional Adversarial Training for Semi-Supervised Domain Adaptation [[IJCAI2020]](https://www.ijcai.org/Proceedings/2020/130)
- Semi-supervised Domain Adaptation via Minimax Entropy [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Saito_Semi-Supervised_Domain_Adaptation_via_Minimax_Entropy_ICCV_2019_paper.pdf) [[Pytorch]](https://github.com/VisionLearningGroup/SSDA_MME)

**Journal**
- Context-guided entropy minimization for semi-supervised domain adaptation [[Neural Networks]](https://doi.org/10.1016/j.neunet.2022.07.011)  [[pytorch]](https://github.com/NingMa-AI/DEEM)

**Arxiv**
- Pred&Guide: Labeled Target Class Prediction for Guiding Semi-Supervised Domain Adaptation [[22 Nov 2022]](https://arxiv.org/abs/2211.11975)
- MiCo: Mixup Co-Training for Semi-Supervised Domain Adaptation [[ 24 Jul 2020]](https://arxiv.org/abs/2007.12684)
- Opposite Structure Learning for Semi-supervised Domain Adaptation [[6 Feb 2020]](https://arxiv.org/abs/2002.02545v1)
- Reducing Domain Gap via Style-Agnostic Networks [[25 Oct 2019]](https://arxiv.org/abs/1910.11645)

## Weakly-Supervised DA

**Conference**
- Towards Accurate and Robust Domain Adaptation under Noisy Environments [[IJCAI2020]](https://www.ijcai.org/Proceedings/2020/0314.pdf)
- Weakly Supervised Open-set Domain Adaptation by Dual-domain Collaboration [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_Weakly_Supervised_Open-Set_Domain_Adaptation_by_Dual-Domain_Collaboration_CVPR_2019_paper.pdf)
- Transferable Curriculum for Weakly-Supervised Domain Adaptation [[AAAI2019]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transferable-curriculum-aaai19.pdf)

**Arxiv**
- Butterfly: Robust One-step Approach towards Wildly-unsupervised Domain Adaptation [[arXiv on 19 May 2019]](https://arxiv.org/abs/1905.07720v1)


## Zero-shot DA
**Conference**
- Collaborative Learning With Disentangled Features for Zero-Shot Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Jhoo_Collaborative_Learning_With_Disentangled_Features_for_Zero-Shot_Domain_Adaptation_ICCV_2021_paper.pdf)
- Zero-Shot Day-Night Domain Adaptation with a Physics Prior [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Lengyel_Zero-Shot_Day-Night_Domain_Adaptation_With_a_Physics_Prior_ICCV_2021_paper.pdf)
- High Resolution Zero-Shot Domain Adaptation of Synthetically Rendered Face Images [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730222.pdf)
- Adversarial Learning for Zero-shot Domain Adaptation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660324.pdf)
- HGNet: Hybrid Generative Network for Zero-shot Domain Adaptation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720052.pdf)
- Zero-shot Domain Adaptation Based on Attribute Information [[ACML2019]](http://proceedings.mlr.press/v101/ishii19a.html)
- Conditional Coupled Generative Adversarial Networks for Zero-Shot Domain Adaptation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Conditional_Coupled_Generative_Adversarial_Networks_for_Zero-Shot_Domain_Adaptation_ICCV_2019_paper.pdf)
- Generalized Zero-Shot Learning with Deep Calibration Network [[NIPS2018]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-calibration-network-nips18.pdf)
- Zero-Shot Deep Domain Adaptation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kuan-Chuan_Peng_Zero-Shot_Deep_Domain_ECCV_2018_paper.pdf)

## One-shot DA
**Conference**
- Adversarial Style Mining for One-Shot Unsupervised Domain Adaptation [[NeurIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/ed265bc903a5a097f61d3ec064d96d2e-Abstract.html) [[Pytorch]](https://github.com/RoyalVane/ASM)
- One-Shot Adaptation of Supervised Deep Convolutional Models [[ICLR Workshop 2014]](https://arxiv.org/abs/1312.6204)

**Arxiv**
- One-Shot Imitation from Observing Humans via Domain-Adaptive Meta-Learning [[arxiv]](https://arxiv.org/abs/1802.01557)

## Few-shot UDA
**Conference**
- Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation
[[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/html/Yue_Prototypical_Cross-Domain_Self-Supervised_Learning_for_Few-Shot_Unsupervised_Domain_Adaptation_CVPR_2021_paper.html) [[Pytorch]](https://github.com/zhengzangw/PCS-FUDA) [[Project]](http://xyue.io/pcs-fuda/)

**Arxiv**
- Cross-domain Self-supervised Learning for Domain Adaptation with Few Source Labels [[arXiv 18 Mar 2020]](https://arxiv.org/pdf/2003.08264.pdf)

## Few-shot DA
**Conference**
- Domain-Adaptive Few-Shot Learning[[WACV2021]](https://openaccess.thecvf.com/content/WACV2021/papers/Zhao_Domain-Adaptive_Few-Shot_Learning_WACV_2021_paper.pdf) [[Pytorch]](https://github.com/dingmyu/DAPN)
- Few-shot Domain Adaptation by Causal Mechanism Transfer [[ICML2020]](https://proceedings.icml.cc/static/paper_files/icml/2020/1121-Paper.pdf) [[Pytorch]](https://github.com/takeshi-teshima/few-shot-domain-adaptation-by-causal-mechanism-transfer)
- Few-Shot Adaptive Faster R-CNN [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Few-Shot_Adaptive_Faster_R-CNN_CVPR_2019_paper.html)
- d-SNE: Domain Adaptation using Stochastic Neighborhood Embedding [[CVPR2019 Oral]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_d-SNE_Domain_Adaptation_Using_Stochastic_Neighborhood_Embedding_CVPR_2019_paper.pdf)
- Few-Shot Adversarial Domain Adaptation [[NIPS2017]](http://papers.nips.cc/paper/7244-few-shot-adversarial-domain-adaptation)

**Arxiv**
- Feature transformation ensemble model with batch spectral regularization for cross-domain few-shot classification [[arXiv 18 May 2020]](https://arxiv.org/pdf/2005.08463.pdf) [[Pytorch]](https://github.com/liubingyuu/FTEM_BSR_CDFSL)
- Ensemble model with batch spectral regularization and data blending for cross-domain few-shot learning with unlabeled data [[arXiv 8 June 2020]](https://arxiv.org/pdf/2006.04323.pdf) [[Pytorch]](https://github.com/123zhen123/BSDB-CDFSL_Track)

## Partial DA
**Conference**
- Implicit Semantic Response Alignment for Partial Domain Adaptation [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/731b03008e834f92a03085ef47061c4a-Abstract.html) [[Pytorch]](https://github.com/implicit-seman-align/Implicit-Semantic-Response-Alignment)
- Adversarial Reweighting for Partial Domain Adaptation [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/7ce3284b743aefde80ffd9aec500e085-Abstract.html)
- A Balanced and Uncertainty-aware Approach for Partial Domain Adaptation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560120.pdf) [[Pytorch]](https://github.com/tim-learn/BA3US)
- Discriminative Partial Domain Adversarial Network [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720630.pdf)
- Selective Transfer With Reinforced Transfer Network for Partial Domain Adaptation [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Selective_Transfer_With_Reinforced_Transfer_Network_for_Partial_Domain_Adaptation_CVPR_2020_paper.pdf)
- Adaptively-Accumulated Knowledge Transfer for Partial Domain Adaptation [[ACM MM2020]](https://dl.acm.org/doi/abs/10.1145/3394171.3413986)
- Multi-Weight Partial Domain Adaptation [[BMVC2019]](https://bmvc2019.org/wp-content/uploads/papers/0406-paper.pdf)
- Learning to Transfer Examples for Partial Domain Adaptation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cao_Learning_to_Transfer_Examples_for_Partial_Domain_Adaptation_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/thuml/ETN)
- Partial Adversarial Domain Adaptation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhangjie_Cao_Partial_Adversarial_Domain_ECCV_2018_paper.pdf) [[Pytorch(Official)]](https://github.com/thuml/PADA)
- Importance Weighted Adversarial Nets for Partial Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Importance_Weighted_Adversarial_CVPR_2018_paper.html) [[Caffe]](https://github.com/hellojing89/weightedGANpartialDA)
- Partial Transfer Learning with Selective Adversarial Networks [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cao_Partial_Transfer_Learning_CVPR_2018_paper.pdf)[[paper weekly]](http://www.paperweekly.site/papers/1388) [[Pytorch(Official) & Caffe(official)]](https://github.com/thuml/SAN)

**Journal**
- Unsupervised Multi-Class Domain Adaptation: Theory, Algorithms, and Practice [[TPAMI2020]](https://arxiv.org/abs/2002.08681) [[PyTroch]](https://github.com/YBZh/MultiClassDA)

**Arxiv**
- Select, Label, and Mix: Learning Discriminative Invariant Feature Representations for Partial Domain Adaptation [[arXiv 06 Dec 2020]](https://arxiv.org/abs/2012.03358)
- Unsupervised Multi-Class Domain Adaptation: Theory, Algorithms, and Practice [[20 Feb 2020]](https://arxiv.org/pdf/2002.08681.pdf) [[PyTroch]](https://github.com/YBZh/MultiClassDA)
- Tackling Partial Domain Adaptation with Self-Supervision [[arXiv 12 Jun 2019]](https://arxiv.org/abs/1906.05199v1)
- Domain Adversarial Reinforcement Learning for Partial Domain Adaptation [[arXiv 10 May 2019]](https://arxiv.org/abs/1905.04094v1)


## Open Set DA
**Conference**
- Towards Novel Target Discovery Through Open-Set Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Jing_Towards_Novel_Target_Discovery_Through_Open-Set_Domain_Adaptation_ICCV_2021_paper.html)
- On the Effectiveness of Image Rotation for Open Set Domain Adaptation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610409.pdf) [[Pytorch]](https://github.com/silvia1993/ROS)
- Multi-Source Open-Set Deep Adversarial Domain Adaptation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710732.pdf)
- Progressive Graph Learning for Open-Set Domain Adaptation [[ICML2020]](https://proceedings.icml.cc/static/paper_files/icml/2020/136-Paper.pdf) [[Pytorch]](https://github.com/BUserName/PGL)
- Joint Partial Optimal Transport for Open Set Domain Adaptation [[IJCAI2020]](https://www.ijcai.org/Proceedings/2020/0352.pdf)
- Exploring Category-Agnostic Clusters for Open-Set Domain Adaptation [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Pan_Exploring_Category-Agnostic_Clusters_for_Open-Set_Domain_Adaptation_CVPR_2020_paper.pdf)
- Towards Inheritable Models for Open-Set Domain Adaptation [[CVPR 2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Kundu_Towards_Inheritable_Models_for_Open-Set_Domain_Adaptation_CVPR_2020_paper.pdf) [[Project]](https://sites.google.com/view/inheritune)
- Attract or Distract: Exploit the Margin of Open Set [[ICCV2019]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Feng_Attract_or_Distract_Exploit_the_Margin_of_Open_Set_ICCV_2019_paper.pdf) [[code]](https://github.com/qy-feng/margin-openset)
- Separate to Adapt: Open Set Domain Adaptation via Progressive Separation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Separate_to_Adapt_Open_Set_Domain_Adaptation_via_Progressive_Separation_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/thuml/Separate_to_Adapt)
- Weakly Supervised Open-set Domain Adaptation by Dual-domain Collaboration [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_Weakly_Supervised_Open-Set_Domain_Adaptation_by_Dual-Domain_Collaboration_CVPR_2019_paper.pdf)
- Learning Factorized Representations for Open-set Domain Adaptation [[ICLR2019]](https://openreview.net/pdf?id=SJe3HiC5KX)
- Open Set Domain Adaptation by Backpropagation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kuniaki_Saito_Adversarial_Open_Set_ECCV_2018_paper.pdf) [[Pytorch(Official)]](https://github.com/ksaito-ut/OPDA_BP) [[Tensorflow]](https://github.com/Mid-Push/Open_set_domain_adaptation) [[Pytorch]](https://github.com/YU1ut/openset-DA)
- Open Set Domain Adaptation [[ICCV2017]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Busto_Open_Set_Domain_ICCV_2017_paper.pdf)

**Journal**
- Open-set domain adaptation by deconfounding domain gaps [[Applied Intelligence 2022]](https://link.springer.com/article/10.1007/s10489-022-03805-9)
- Unsupervised Multi-Class Domain Adaptation: Theory, Algorithms, and Practice [[TPAMI2020]](https://arxiv.org/abs/2002.08681) [[PyTroch]](https://github.com/YBZh/MultiClassDA)
- Adversarial Network with Multiple Classifiers for Open Set Domain Adaptation [[IEEE TMM]](https://arxiv.org/abs/2007.00384) [[Pytorch]](https://github.com/tasfia/DAMC)

**Arxiv**
- Collaborative Training of Balanced Random Forests for Open Set Domain Adaptation [[10 Feb 2020]](https://arxiv.org/abs/2002.03642v1)
- Known-class Aware Self-ensemble for Open Set Domain Adaptation [[3 May 2019]](https://arxiv.org/abs/1905.01068v1)

## Universal DA
**Conference**
- Subsidiary Prototype Alignment for Universal Domain Adaptation [[NeurIPS2022]](https://openreview.net/forum?id=5kThooa07pf) [[Project Page]](https://sites.google.com/view/spa-unida)
- OVANet: One-vs-All Network for Universal Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Saito_OVANet_One-vs-All_Network_for_Universal_Domain_Adaptation_ICCV_2021_paper.html)
- Active Universal Domain Adaptation [[ICCV 2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Ma_Active_Universal_Domain_Adaptation_ICCV_2021_paper.pdf)
- Domain Consensus Clustering for Universal Domain Adaptation [[CVPR 2021]](http://reler.net/papers/guangrui_cvpr2021.pdf) [[Pytorch]](https://github.com/Solacex/Domain-Consensus-Clustering)
- Divergence Optimization for Noisy Universal Domain Adaptation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yu_Divergence_Optimization_for_Noisy_Universal_Domain_Adaptation_CVPR_2021_paper.pdf)
- Universal Domain Adaptation through Self Supervision [[NeurIPS 2020]](https://papers.nips.cc/paper/2020/hash/bb7946e7d85c81a9e69fee1cea4a087c-Abstract.html) [[Pytorch]](https://github.com/VisionLearningGroup/DANCE)
- Learning to Detect Open Classes for Universal Domain Adaptation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600562.pdf) [[code]](https://github.com/thuml/Calibrated-Multiple-Uncertainties)
- Universal Source-Free Domain Adaptation [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Kundu_Universal_Source-Free_Domain_Adaptation_CVPR_2020_paper.pdf) [[Project]](https://sites.google.com/view/usfda-cvpr2020)
- Universal Domain Adaptation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/You_Universal_Domain_Adaptation_CVPR_2019_paper.pdf)  [[Pytorch]](https://github.com/thuml/Universal-Domain-Adaptation)


**Journal**

- Universal Model Adaptation by Style Augmented Open-set Consistency [[Applied Intelligence 2023]](https://link.springer.com/article/10.1007/s10489-023-04731-0)

**Arxiv**
- Universal Multi-Source Domain Adaptation [[5 Nov 2020]](https://arxiv.org/abs/2011.02594)
- A Sample Selection Approach for Universal Domain Adaptation [[14 Jan 2020]](https://arxiv.org/abs/2001.05071v1)




## Open Compound DA
**Conference**
- Discover, Hallucinate, and Adapt: Open Compound Domain Adaptation for Semantic Segmentation [[NeurIPS2020]](https://proceedings.neurips.cc/paper/2020/file/7a9a322cbe0d06a98667fdc5160dc6f8-Paper.pdf)
- Open Compound Domain Adaptation [[CVRP2020 Oral]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Open_Compound_Domain_Adaptation_CVPR_2020_paper.pdf) [[Pytorch]](https://github.com/zhmiao/OpenCompoundDomainAdaptation-OCDA)

**Journal**
- Source-Free Open Compound Domain Adaptation in Semantic Segmentation [[TCSVT 2022]](https://ieeexplore.ieee.org/document/9785619)

## Multi Source DA

**Conference**
- Confident Anchor-Induced Multi-Source Free Domain Adaptation [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/168908dd3227b8358eababa07fcaf091-Abstract.html) [[code is coming soon]](https://github.com/Learning-group123/CAiDA)
- mDALU: Multi-Source Domain Adaptation and Label Unification With Partial Datasets [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Gong_mDALU_Multi-Source_Domain_Adaptation_and_Label_Unification_With_Partial_Datasets_ICCV_2021_paper.html)
- STEM: An Approach to Multi-Source Domain Adaptation With Guarantees [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Nguyen_STEM_An_Approach_to_Multi-Source_Domain_Adaptation_With_Guarantees_ICCV_2021_paper.html)
- T-SVDNet: Exploring High-Order Prototypical Correlations for Multi-Source Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Li_T-SVDNet_Exploring_High-Order_Prototypical_Correlations_for_Multi-Source_Domain_Adaptation_ICCV_2021_paper.html)
- Multi-Source Domain Adaptation for Object Detection [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Yao_Multi-Source_Domain_Adaptation_for_Object_Detection_ICCV_2021_paper.html)
- Information-Theoretic Regularization for Multi-Source Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Park_Information-Theoretic_Regularization_for_Multi-Source_Domain_Adaptation_ICCV_2021_paper.html)
- Partial Feature Selection and Alignment for Multi-Source Domain Adaptation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Fu_Partial_Feature_Selection_and_Alignment_for_Multi-Source_Domain_Adaptation_CVPR_2021_paper.pdf)
- Wasserstein Barycenter for Multi-Source Domain Adaptation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Montesuma_Wasserstein_Barycenter_for_Multi-Source_Domain_Adaptation_CVPR_2021_paper.pdf) [[Code]](https://github.com/eddardd/WBTransport)
- Unsupervised Multi-source Domain Adaptation Without Access to Source Data [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ahmed_Unsupervised_Multi-Source_Domain_Adaptation_Without_Access_to_Source_Data_CVPR_2021_paper.pdf)
- Dynamic Transfer for Multi-Source Domain Adaptation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Dynamic_Transfer_for_Multi-Source_Domain_Adaptation_CVPR_2021_paper.pdf) [[Pytorch]](https://github.com/liyunsheng13/DRT)
- Multi-Source Domain Adaptation with Collaborative Learning for Semantic Segmentation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/He_Multi-Source_Domain_Adaptation_With_Collaborative_Learning_for_Semantic_Segmentation_CVPR_2021_paper.pdf)
- MOST: Multi-Source Domain Adaptation via Optimal Transport for Student-Teacher Learning [[UAI2021]](https://auai.org/uai2021/pdf/uai2021.106.pdf)
- Meta Self-Learning for Multi-Source Domain Adaptation: A Benchmark [[ICCV Workshop 2021]](https://arxiv.org/abs/2108.10840) [[Pytorch]](https://github.com/bupt-ai-cz/Meta-SelfLearning)
- Your Classifier can Secretly Suffice Multi-Source Domain Adaptation [[NeurIPS 2020]](https://papers.nips.cc/paper/2020/file/3181d59d19e76e902666df5c7821259a-Paper.pdf) [[Project]](https://sites.google.com/view/simpal)
- Multi-Source Open-Set Deep Adversarial Domain Adaptation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710732.pdf)
- Online Meta-Learning for Multi-Source and Semi-Supervised Domain Adaptation [[ECCV2020]](https://arxiv.org/abs/2004.04398)
- Multi-Source Open-Set Deep Adversarial Domain Adaptation [[ECCV2020]](https://dipeshtamboli.github.io/blog/2020/Multi-Source-Open-Set-Deep-Adversarial-Domain-Adaptation/)
- Curriculum Manager for Source Selection in Multi-Source Domain Adaptation [[ECCV2020]](https://arxiv.org/abs/2007.01261v1)
- Domain Aggregation Networks for Multi-Source Domain Adaptation [[ICML2020]](https://proceedings.icml.cc/static/paper_files/icml/2020/6292-Paper.pdf)
- Learning to Combine: Knowledge Aggregation for Multi-Source Domain Adaptation [[ECCV2020]](https://github.com/ChrisAllenMing/LtC-MSDA) [[Pytorch]](https://github.com/ChrisAllenMing/LtC-MSDA)
- Multi-Source Domain Adaptation for Text Classification via DistanceNet-Bandits [[AAAI2020]](https://arxiv.org/abs/2001.04362v2)
- Adversarial Training Based Multi-Source Unsupervised Domain Adaptation for Sentiment Analysis [[AAAI2020]](https://arxiv.org/pdf/2006.05602.pdf)
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
- Graphical Modeling for Multi-Source Domain Adaptation [[TPAMI 2022]](https://ieeexplore.ieee.org/abstract/document/9767755) [[Pytorch]](https://github.com/Francis0625/Graphical-Modeling-for-Multi-Source-Domain-Adaptation)
- Unsupervised sentiment analysis by transferring multi-source knowledge[[Cognitive Computation]](https://arxiv.org/pdf/2105.11902.pdf)
- A survey of multi-source domain adaptation [[Information Fusion]](https://www.sciencedirect.com/science/article/pii/S1566253514001316)

**Arxiv**
- Mutual learning network for multi-source domain adaptation [[arXiv]](https://arxiv.org/pdf/2003.12944)
- Domain Adaptive Ensemble Learning [[arXiv]](https://arxiv.org/abs/2003.07325)
- Multi-Source Domain Adaptation and Semi-Supervised Domain Adaptation with Focus on Visual Domain Adaptation Challenge 2019 [[14 Oct 2019]](https://arxiv.org/abs/1910.03548)

## Multi Target DA
**Conference**
- CoNMix for Source-free Single and Multi-target Domain Adaptation [[WACV2022]](https://openaccess.thecvf.com/content/WACV2023/html/Kumar_CoNMix_for_Source-Free_Single_and_Multi-Target_Domain_Adaptation_WACV_2023_paper.html) [[Pytorch]](https://github.com/vcl-iisc/CoNMix)
- Curriculum Graph Co-Teaching for Multi-Target Domain Adaptation [[CVPR2021]](https://arxiv.org/abs/2104.00808v1) [[Pytorch]](https://openaccess.thecvf.com/content/CVPR2021/papers/Roy_Curriculum_Graph_Co-Teaching_for_Multi-Target_Domain_Adaptation_CVPR_2021_paper.pdf)
- Multi-Target Domain Adaptation with Collaborative Consistency Learning [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Isobe_Multi-Target_Domain_Adaptation_With_Collaborative_Consistency_Learning_CVPR_2021_paper.pdf)

**Arxiv**
- Unsupervised Multi-Target Domain Adaptation: An Information Theoretic Approach [[arXiv]](https://arxiv.org/abs/1810.11547v1)

## Incremental DA
**Conference**
- Lifelong Domain Adaptation via Consolidated Internal Distribution [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/5caf41d62364d5b41a893adc1a9dd5d4-Abstract.html)
- Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Volpi_Continual_Adaptation_of_Visual_Representations_via_Domain_Randomization_and_Meta-Learning_CVPR_2021_paper.pdf)
- ConDA: Continual Unsupervised Domain Adaptation [[CVPR2021]](https://arxiv.org/abs/2103.11056v1)
- Gradient Regularized Contrastive Learning for Continual Domain Adaptation [[AAAI2021]](https://arxiv.org/abs/2103.12294v1)
- Gradual Domain Adaptation without Indexed Intermediate Domains [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/45017f6511f91be700fda3d118034994-Abstract.html)
- Learning to Adapt to Evolving Domains [[NeurIPS 2020]](https://proceedings.neurips.cc/paper/2020/file/fd69dbe29f156a7ef876a40a94f65599-Paper.pdf) [[Pytorch]](https://github.com/Liuhong99/EAML)
- Class-Incremental Domain Adaptation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580052.pdf)
- Incremental Adversarial Domain Adaptation for Continually Changing Environments [[ICRA2018]](https://arxiv.org/abs/1712.07436)
- Continuous Manifold based Adaptation for Evolving Visual Domains [[CVPR2014]](https://people.eecs.berkeley.edu/~jhoffman/papers/Hoffman_CVPR2014.pdf)

## Multi Step DA
**Arxiv**
- Adversarial Domain Adaptation for Stance Detection [[arXiv]](https://arxiv.org/abs/1902.02401)
- Ensemble Adversarial Training: Attacks and Defenses [[arXiv]](https://arxiv.org/abs/1705.07204)

**Conference**
- Distant domain transfer learning [[AAAI2017]](http://www.ntu.edu.sg/home/sinnopan/publications/[AAAI17]Distant%20Domain%20Transfer%20Learning.pdf)

## Heterogeneous DA
**Conference**
- Domain Adaptive Classification on Heterogeneous Information Networks [[IJCAI2020]](https://www.ijcai.org/Proceedings/2020/0196.pdf)
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

## Continuously Indexed DA
**Conference**
- Continuously Indexed Domain Adaptation [[ICML 2020]](http://wanghao.in/paper/ICML20_CIDA.pdf) [[Pytorch]](https://github.com/hehaodele/CIDA) [[Project Page]](https://github.com/hehaodele/CIDA/blob/master/README.md) [[Video]](https://www.youtube.com/watch?v=KtZPSCD-WhQ)

## Source Free DA
**Conference**
- Domain Adaptation with Adversarial Training on Penultimate Activations [[AAAI2023]](https://ojs.aaai.org/index.php/AAAI/article/view/26185) [[Pytorch]](https://github.com/tsun/APA)
- Source-free Domain Adaptive Human Pose Estimation [[ICCV2023]](https://arxiv.org/abs/2308.03202)[[Pytorch]](https://github.com/davidpengucf/SFDAHPE)
- RAIN: RegulArization on Input and Network for Black-Box Domain Adaptation [[IJCAI2023]](https://www.ijcai.org/proceedings/2023/458) [[Pytorch]](https://github.com/davidpengucf/RAIN)
- CoNMix for Source-free Single and Multi-target Domain Adaptation [[WACV2022]](https://openaccess.thecvf.com/content/WACV2023/html/Kumar_CoNMix_for_Source-Free_Single_and_Multi-Target_Domain_Adaptation_WACV_2023_paper.html) [[Pytorch]](https://github.com/vcl-iisc/CoNMix)
- Source-free Video Domain Adaptation by Learning Temporal Consistency for Action Recognition [[ECCV2022]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940144.pdf) [[Pytorch]](https://github.com/xuyu0010/ATCoN) [[Project]](https://xuyu0010.github.io/sfvda.html)
- Concurrent Subsidiary Supervision for Unsupervised Source-Free Domain Adaptation [[ECCV2022]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/912_ECCV_2022_paper.php) [[Project Page]](https://sites.google.com/view/sticker-sfda)
- Balancing Discriminability and Transferability for Source-Free Domain Adaptation [[ICML2022]](https://proceedings.mlr.press/v162/kundu22a.html) [[Project Page]](https://sites.google.com/view/mixup-sfda)
- Source-free Domain Adaptation via Avatar Prototype Generation and Adaptation [[IJCAI2021]](https://arxiv.org/abs/2106.15326) [[Pytorch]](https://github.com/SCUT-AILab/CPGA)
- Confident Anchor-Induced Multi-Source Free Domain Adaptation [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/168908dd3227b8358eababa07fcaf091-Abstract.html) [[Pytorch]](https://github.com/Learning-group123/CAiDA)
- Model Adaptation: Historical Contrastive Learning for Unsupervised Domain Adaptation without Source Data [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/1dba5eed8838571e1c80af145184e515-Abstract.html) [[Pytorch]](https://github.com/jxhuang0508/HCL)
- Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/f5deaeeae1538fb6c45901d524ee2f98-Abstract.html) [[Pytorch]](https://github.com/Albert0147/SFDA_neighbors)
- Unsupervised Domain Adaptation of Black-Box Source Models [[BMVC2021]](https://www.bmvc2021-virtualconference.com/assets/papers/0404.pdf)[[Pytorch]](https://github.com/zhjscut/IterLNL)
- Generalize Then Adapt: Source-Free Domain Adaptive Semantic Segmentation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Kundu_Generalize_Then_Adapt_Source-Free_Domain_Adaptive_Semantic_Segmentation_ICCV_2021_paper.html) [[Project]](https://sites.google.com/view/sfdaseg)
- Generalized Source-free Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Generalized_Source-Free_Domain_Adaptation_ICCV_2021_paper.pdf) [[Pytorch]](https://github.com/Albert0147/G-SFDA)
- Adaptive Adversarial Network for Source-free Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xia_Adaptive_Adversarial_Network_for_Source-Free_Domain_Adaptation_ICCV_2021_paper.pdf) [[Pytorch]](https://github.com/HaifengXia/SFDA)
- Visualizing Adapted Knowledge in Domain Transfer [[CVPR2021]](https://arxiv.org/abs/2104.10602) [[Pytorch]](https://github.com/hou-yz/DA_visualization)
- Unsupervised Multi-source Domain Adaptation Without Access to Source Data [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ahmed_Unsupervised_Multi-Source_Domain_Adaptation_Without_Access_to_Source_Data_CVPR_2021_paper.pdf) [[Pytorch]](https://github.com/driptaRC/DECISION)
- Source-Free Domain Adaptation for Semantic Segmentation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Source-Free_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2021_paper.pdf)
- Domain Impression: A Source Data Free Domain Adaptation Method [[WACV2021]](https://openaccess.thecvf.com/content/WACV2021/papers/Kurmi_Domain_Impression_A_Source_Data_Free_Domain_Adaptation_Method_WACV_2021_paper.pdf) [[Project]](https://delta-lab-iitk.github.io/SFDA/)
- Model Adaptation: Unsupervised Domain Adaptation Without Source Data [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Model_Adaptation_Unsupervised_Domain_Adaptation_Without_Source_Data_CVPR_2020_paper.pdf)
- Universal Source-Free Domain Adaptation [[CVPR2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kundu_Universal_Source-Free_Domain_Adaptation_CVPR_2020_paper.pdf) [[Project]](https://sites.google.com/view/usfda-cvpr2020)
- Towards Inheritable Models for Open-Set Domain Adaptation [[CVPR2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kundu_Towards_Inheritable_Models_for_Open-Set_Domain_Adaptation_CVPR_2020_paper.pdf) [[Project]](https://sites.google.com/view/inheritune)
- Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation [[ICML2020]](http://proceedings.mlr.press/v119/ishida20a.html) [[Pytorch]](https://github.com/tim-learn/SHOT)

**Arxiv**
- Learning Invariant Representation with Consistency and Diversity for Semi-supervised Source Hypothesis Transfer[[7 Jul 2021]](https://arxiv.org/abs/2107.03008)[[Pytorch]](https://github.com/Wang-xd1899/SSHT)
- Source Data-absent Unsupervised Domain Adaptation through Hypothesis Transfer and Labeling Transfer [[14 Dec 2020]](https://arxiv.org/abs/2012.07297) [[Pytorch]](https://github.com/tim-learn/SHOT-plus)

## Active DA
**Conference**
- Local Context-Aware Active Domain Adaptation [[ICCV2023]](https://arxiv.org/abs/2208.12856) [[Pytorch]](https://github.com/tsun/LADA)
- Reducing Annotation Effort by Identifying and Labeling Contextually Diverse Classes for Semantic Segmentation Under Domain Shift [[WACV2023]](https://openaccess.thecvf.com/content/WACV2023/papers/Agarwal_Reducing_Annotation_Effort_by_Identifying_and_Labeling_Contextually_Diverse_Classes_WACV_2023_paper.pdf)
- Towards Fewer Annotations: Active Learning via Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_Towards_Fewer_Annotations_Active_Learning_via_Region_Impurity_and_Prediction_CVPR_2022_paper.pdf)[[Pytorch]](https://github.com/BIT-DA/RIPU)
- Active Learning for Domain Adaptation: An Energy-based Approach [[AAAI2022]](ttps://arxiv.org/abs/2112.01406)[[Pytorch]](https://github.com/BIT-DA/EADA)
- Multi-Anchor Active Domain Adaptation for Semantic Segmentation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Ning_Multi-Anchor_Active_Domain_Adaptation_for_Semantic_Segmentation_ICCV_2021_paper.html)
- Active Domain Adaptation via Clustering Uncertainty-Weighted Embeddings [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Prabhu_Active_Domain_Adaptation_via_Clustering_Uncertainty-Weighted_Embeddings_ICCV_2021_paper.html)
- Active Universal Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Ma_Active_Universal_Domain_Adaptation_ICCV_2021_paper.html)
- S3VAADA: Submodular Subset Selection for Virtual Adversarial Active Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Rangwani_S3VAADA_Submodular_Subset_Selection_for_Virtual_Adversarial_Active_Domain_Adaptation_ICCV_2021_paper.pdf)
- Transferable Query Selection for Active Domain Adaptation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Fu_Transferable_Query_Selection_for_Active_Domain_Adaptation_CVPR_2021_paper.pdf)

## Generalized Domain Adaptation
**Conference**
- Generalized Domain Adaptation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Mitsuzumi_Generalized_Domain_Adaptation_CVPR_2021_paper.pdf)

## Model Selection
- Addressing Parameter Choice Issues in Unsupervised Domain Adaptation by Aggregation [[ICLR2023ORAL]](https://openreview.net/forum?id=M95oDwJXayG) [[Pytorch]](https://github.com/Xpitfire/iwa)
- The Balancing Principle for Parameter Choice in Distance-Regularized Domain Adaptation [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/ae0909a324fb2530e205e52d40266418-Abstract.html) [[Pytorch]](https://github.com/xpitfire/bpda)
- Towards Accurate Model Selection in Deep Unsupervised Domain Adaptation [[ICML2019]](http://proceedings.mlr.press/v97/you19a/you19a.pdf) [[Pytorch]](https://github.com/thuml/Deep-Embedded-Validation)

## Other Transfer Learning Paradigms
### Domain Generalization

**Conference**
- Adapting to Distribution Shift by Visual Domain Prompt Generation [[ICLR2024]](https://arxiv.org/pdf/2405.02797) [[Pytorch]](https://github.com/Guliisgreat/VDPG)
- Test-Time Domain Adaptation by Learning Domain-Aware Batch Normalization [[AAAI2024 (Oral)]](https://arxiv.org/pdf/2312.10165)
- A Dual-Augmentor Framework for Domain Generalization in 3D Human Pose Estimation [[CVPR 2024]](https://arxiv.org/abs/2403.11310) [[Pytorch]](https://github.com/davidpengucf/DAF-DG)
- Generalization by Adaptation: Diffusion-Based Domain Extension for Domain-Generalized Semantic Segmentation [[WACV 2024]](https://arxiv.org/abs/2312.01850) [[Pytorch]](https://github.com/JNiemeijer/DIDEX)
- Topology-aware Robust Optimization for Out-of-Distribution Generalization [[ICLR 2023]](https://arxiv.org/pdf/2307.13943) [[Pytorch]](https://github.com/joffery/TRO)
- A Re-Parameterized Vision Transformer (ReVT) for Domain-Generalized Semantic Segmentation [[ICCV Workshop 2023]](https://arxiv.org/abs/2308.13331) [[Pytorch]](https://github.com/ifnspaml/revt-domain-generalization)
- Weight Averaging Improves Knowledge Distillation under Domain Shift [[ICCV Workshop 2023]](https://arxiv.org/abs/2309.11446) [[Pytorch]](https://github.com/vorobeevich/distillation-in-dg)
- Adaptive Texture Filtering for Single-Domain Generalized Segmentation [[AAAI2023 oral]](https://arxiv.org/abs/2303.02943) [[Pytorch]](https://github.com/leelxh/Adaptive-Texture-Filtering-for-Single-Domain-Generalized-Segmentation)
- PromptStyler: Prompt-driven Style Generation for Source-free Domain Generalization [[ICCV2023]](https://arxiv.org/abs/2307.15199) [[Project]](https://promptstyler.github.io/)
- Sparse Mixture-of-Experts are Domain Generalizable Learners [[ICLR2023(Oral)]](https://openreview.net/forum?id=RecZ9nB9Q4) [[Pytorch]](https://github.com/Luodian/Generalizable-Mixture-of-Experts)
- Meta-DMoE: Adapting to Domain Shift by Meta-Distillation from Mixture-of-Experts [[NeruIPS2022]](https://arxiv.org/pdf/2210.03885.pdf) [[Pytorch]](https://github.com/n3il666/Meta-DMoE)
- Style-Hallucinated Dual Consistency Learning for Domain Generalized Semantic Segmentation [[ECCV 2022]](https://arxiv.org/pdf/2204.02548.pdf) [[Pytorch]](https://github.com/HeliosZhao/SHADE)
- Learning to Generalize Unseen Domains via Memory-based Multi-Source Meta-Learning for Person Re-Identification [[CVPR 2021]](https://arxiv.org/pdf/2012.00417.pdf) [[Pytorch]](https://github.com/HeliosZhao/M3L)
- Domain Generalization via Inference-time Label-Preserving Target Projections [[CVPR2021]](https://arxiv.org/abs/2103.01134) [[Pytorch]](https://github.com/peterDan8/InferenceTimeDG)
- Domain Generalization via Entropy Regularization [[NeurIPS2020]](https://papers.nips.cc/paper/2020/file/b98249b38337c5088bbc660d8f872d6a-Paper.pdf) [[Pytorch]](https://github.com/sshan-zhao/DG_via_ER)
- Domain Generalization for Medical Imaging Classification with Linear-Dependency Regularization [[NeurIPS2020]](https://arxiv.org/abs/2009.12829)
- Learning to Learn with Variational Information Bottleneck for Domain Generalization [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550205.pdf)
- Self-Challenging Improves Cross-Domain Generalization [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470120.pdf) [[Pytorch]](https://github.com/DeLightCMU/RSC)
- Learning from Extrinsic and Intrinsic Supervisions for Domain Generalization [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540154.pdf) [[Pytorch]](https://github.com/emma-sjwang/EISNet)
- Learning to Balance Specificity and Invariance for In and Out of Domain Generalization [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540290.pdf) [[Pytorch]](https://github.com/prithv1/DMG)
- Learning to Generate Novel Domains for Domain Generalization [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610545.pdf)
- Learning to Optimize Domain Specific Normalization for Domain Generalization [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670069.pdf)
- Towards Recognizing Unseen Categories in Unseen Domains [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680460.pdf) [[Pytorch]](https://github.com/mancinimassimiliano/CuMix)
- Efficient Domain Generalization via Common-Specific Low-Rank Decomposition [[ICML2020]](https://proceedings.icml.cc/static/paper_files/icml/2020/4649-Paper.pdf) [[Pytorch]](https://github.com/vihari/csd)
- Learning to Learn Single Domain Generalization [[CVPR2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qiao_Learning_to_Learn_Single_Domain_Generalization_CVPR_2020_paper.pdf) [[Pytorch]](https://github.com/joffery/M-ADA)
- Generalized Convolutional Forest Networks for Domain Generalization and Visual Recognition [[ICLR2020]](https://openreview.net/forum?id=H1lxVyStPH)
- Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation [[ICLR2020]](https://openreview.net/forum?id=SJl5Np4tPr)
- Domain Generalization Using a Mixture of Multiple Latent Domains [[AAAI2020]](https://arxiv.org/abs/1911.07661v1) [[Pytorch]](https://github.com/mil-tokyo/dg_mmld)
- Deep Domain-Adversarial Image Generation for Domain Generalisation [[Paper]](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-ZhouK.2138.pdf) [[Pytorch]](https://github.com/KaiyangZhou/Dassl.pytorch)
- Domain Generalization via Model-Agnostic Learning of Semantic Features [[NeurIPS2019]](https://papers.nips.cc/paper/8873-domain-generalization-via-model-agnostic-learning-of-semantic-features) [[Tensorflow]](https://github.com/biomedia-mira/masf)
- Episodic Training for Domain Generalization [[ICCV2019 Oral]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Episodic_Training_for_Domain_Generalization_ICCV_2019_paper.pdf) [Pytorch]](https://github.com/HAHA-DL/Episodic-DG)
- Feature-Critic Networks for Heterogeneous Domain Generalization [[ICML2019]](http://proceedings.mlr.press/v97/li19l/li19l.pdf) [[Pytorch]](https://github.com/liyiying/Feature_Critic)
- Domain Generalization by Solving Jigsaw Puzzles [[CVPR2019 Oral]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Carlucci_Domain_Generalization_by_Solving_Jigsaw_Puzzles_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/fmcarlucci/JigenDG)
- MetaReg: Towards Domain Generalization using Meta-Regularization [[NIPS2018]](https://papers.nips.cc/paper/7378-metareg-towards-domain-generalization-using-meta-regularization)
- Deep Domain Generalization via Conditional Invariant Adversarial Networks [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.pdf)
- Domain Generalization with Adversarial Feature Learning [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Domain_Generalization_With_CVPR_2018_paper.pdf)

**Journal**
- Domain Generalization for Regression [[IntellManuf2020]](https://link.springer.com/article/10.1007/s10845-019-01499-4)
- Correlation-aware Adversarial Domain Adaptation and Generalization [[Pattern Recognition(2019)]](https://arxiv.org/abs/1911.12983v1) [[code]](https://github.com/mahfujur1/CA-DA-DG)

**Arxiv**
- Adversarial Pyramid Network for Video Domain Generalization [[8 Dec 2019]](https://arxiv.org/abs/1912.03716)
- Towards Shape Biased Unsupervised Representation Learning for Domain Generalization [[18 Sep 2019]](https://arxiv.org/abs/1909.08245v1)
- A Generalization Error Bound for Multi-class Domain Generalization [[24 May 2019]](https://arxiv.org/abs/1905.10392v1)
- Adversarial Invariant Feature Learning with Accuracy Constraint for Domain Generalization [[29 Apr 2019]](https://arxiv.org/abs/1904.12543v1)
- Beyond Domain Adaptation: Unseen Domain Encapsulation via Universal Non-volume Preserving Models [[9 Dec 2018]](https://arxiv.org/abs/1812.03407v1)

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

**Survey**
- Unsupervised Domain Adaptation of Object Detectors: A Survey [[Arxiv 27 May 2021]](https://arxiv.org/abs/2105.13502)
  
**Conference**
- Supervision Interpolation via LossMix: Generalizing Mixup for Object Detection and Beyond [[AAAI2024]](https://arxiv.org/abs/2303.10343)
- Instance Relation Graph Guided Source-Free Domain Adaptive Object Detection [[CVPR2023]](https://arxiv.org/abs/2203.15793) [[Project]](https://viudomain.github.io/irg-sfda-web/)
- Towards Online Domain Adaptive Object Detection [[WACV2023]](https://arxiv.org/abs/2204.05289) [[https://github.com/Vibashan/online-da]]
- Mixture of Teacher Experts for Source-Free Domain Adaptive Object Detection [[ICIP2022]](https://ieeexplore.ieee.org/document/9897795)
- Towards Robust Adaptive Object Detection under Noisy Annotations [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Towards_Robust_Adaptive_Object_Detection_Under_Noisy_Annotations_CVPR_2022_paper.pdf) [[PyTorch]](https://github.com/CityU-AIM-Group/NLTE)
- H<sup>2</sup>FA R-CNN: Holistic and Hierarchical Feature Alignment for Cross-Domain Weakly Supervised Object Detection [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_H2FA_R-CNN_Holistic_and_Hierarchical_Feature_Alignment_for_Cross-Domain_Weakly_CVPR_2022_paper.pdf) [[PyTorch]](https://github.com/XuYunqiu/H2FA_R-CNN) [[PaddlePaddle]](https://github.com/XuYunqiu/H2FA_R-CNN/tree/ppdet)
- Cross-Domain Adaptive Teacher for Object Detection [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection_CVPR_2022_paper.pdf) [[Project]](https://yujheli.github.io/projects/adaptiveteacher.html) [[PyTorch]](https://github.com/facebookresearch/adaptive_teacher)
- Task-specific Inconsistency Alignment for Domain Adaptive Object Detection [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_Task-Specific_Inconsistency_Alignment_for_Domain_Adaptive_Object_Detection_CVPR_2022_paper.pdf) [[PyTorch]](https://github.com/MCG-NJU/TIA)
- SIGMA: Semantic-complete Graph Matching for Domain Adaptive Object Detection [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_SIGMA_Semantic-Complete_Graph_Matching_for_Domain_Adaptive_Object_Detection_CVPR_2022_paper.pdf) [[PyTorch]](https://github.com/CityU-AIM-Group/SIGMA)
- Single-Domain Generalized Object Detection in Urban Scene via Cyclic-Disentangled Self-Distillation [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Single-Domain_Generalized_Object_Detection_in_Urban_Scene_via_Cyclic-Disentangled_Self-Distillation_CVPR_2022_paper.pdf)
- Target-Relevant Knowledge Preservation for Multi-Source Domain Adaptive Object Detection [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Target-Relevant_Knowledge_Preservation_for_Multi-Source_Domain_Adaptive_Object_Detection_CVPR_2022_paper.pdf)
- Cross Domain Object Detection by Target-Perceived Dual Branch Distillation [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Cross_Domain_Object_Detection_by_Target-Perceived_Dual_Branch_Distillation_CVPR_2022_paper.pdf)
- Decoupled Adaptation for Cross-Domain Object Detection [[ICLR2022]](https://openreview.net/pdf?id=VNqaB1g9393) [[PyTorch]](https://github.com/thuml/Decoupled-Adaptation-for-Cross-Domain-Object-Detection)
- SCAN: Cross Domain Object Detection with Semantic Conditioned Adaptation [[AAAI2022]](https://www.aaai.org/AAAI22Papers/AAAI-902.LiW.pdf) [[PyTorch]](https://github.com/CityU-AIM-Group/SCAN)
- SSAL: Synergizing between Self-Training and Adversarial Learning for Domain Adaptive Object Detection [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/file/c0cccc24dd23ded67404f5e511c342b0-Paper.pdf) [[Project]](http://im.itu.edu.pk/synergizing-domain-adaptation/)
- Multi-Source Domain Adaptation for Object Detection [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Yao_Multi-Source_Domain_Adaptation_for_Object_Detection_ICCV_2021_paper.html)
- Knowledge Mining and Transferring for Domain Adaptive Object Detection [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Tian_Knowledge_Mining_and_Transferring_for_Domain_Adaptive_Object_Detection_ICCV_2021_paper.html)
- Dual Bipartite Graph Learning: A General Approach for Domain Adaptive Object Detection [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Dual_Bipartite_Graph_Learning_A_General_Approach_for_Domain_Adaptive_ICCV_2021_paper.pdf)
- Seeking Similarities over Differences: Similarity-based Domain Alignment for Adaptive Object Detection [[ICCV2021]](https://arxiv.org/pdf/2110.01428.pdf)
- Informative and Consistent Correspondence Mining for Cross-Domain Weakly Supervised Object Detection [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Hou_Informative_and_Consistent_Correspondence_Mining_for_Cross-Domain_Weakly_Supervised_Object_CVPR_2021_paper.pdf)
- MeGA-CDA: Memory Guided Attention for Category-Aware Unsupervised Domain Adaptive Object Detection [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/VS_MeGA-CDA_Memory_Guided_Attention_for_Category-Aware_Unsupervised_Domain_Adaptive_Object_CVPR_2021_paper)
- SRDAN: Scale-aware and Range-aware Domain Adaptation Network
for Cross-dataset 3D Object Detection [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_SRDAN_Scale-Aware_and_Range-Aware_Domain_Adaptation_Network_for_Cross-Dataset_3D_CVPR_2021_paper.pdf)
- I3Net: Implicit Instance-Invariant Network for Adapting One-Stage Object Detectors [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_I3Net_Implicit_Instance-Invariant_Network_for_Adapting_One-Stage_Object_Detectors_CVPR_2021_paper.pdf)
- RPN Prototype Alignment for Domain Adaptive Object Detector [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_RPN_Prototype_Alignment_for_Domain_Adaptive_Object_Detector_CVPR_2021_paper.pdf)
- ST3D: Self-training for Unsupervised Domain Adaptation on 3D ObjectDetection [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_ST3D_Self-Training_for_Unsupervised_Domain_Adaptation_on_3D_Object_Detection_CVPR_2021_paper.pdf)
- Domain-Specific Suppression for Adaptive Object Detection [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Domain-Specific_Suppression_for_Adaptive_Object_Detection_CVPR_2021_paper.pdf)
- Unbiased Mean Teacher for Cross-Domain Object Detection [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Deng_Unbiased_Mean_Teacher_for_Cross-Domain_Object_Detection_CVPR_2021_paper.pdf)
- YOLO in the Dark - Domain Adaptation Method for Merging Multiple Models [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660341.pdf)
- Collaborative Training between Region Proposal Localization and Classification for Domain Adaptive Object Detection [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630086.pdf)
- One-Shot Unsupervised Cross-Domain Detection [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610715.pdf)
- Every Pixel Matters: Center-aware Feature Alignment for Domain Adaptive Object Detector [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540698.pdf)
- Adapting Object Detectors with Conditional Domain Normalization [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560392.pdf)
- Prior-based Domain Adaptive Object Detection for Hazy and Rainy Conditions [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590749.pdf)
- Domain Adaptive Object Detection via Asymmetric Tri-way Faster-RCNN [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690307.pdf)
- Cross-domain Object Detection through Coarse-to-Fine Feature Adaptation [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Cross-domain_Object_Detection_through_Coarse-to-Fine_Feature_Adaptation_CVPR_2020_paper.pdf)
- Harmonizing Transferability and Discriminability for Adapting Object Detectors [[CVPR2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Harmonizing_Transferability_and_Discriminability_for_Adapting_Object_Detectors_CVPR_2020_paper.pdf) [[code]](https://github.com/chaoqichen/HTCN)
- Exploring Categorical Regularization for Domain Adaptive Object Detection [[CVPR2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Exploring_Categorical_Regularization_for_Domain_Adaptive_Object_Detection_CVPR_2020_paper.pdf) [[code]](https://github.com/Megvii-Nanjing/CR-DA-DET)
- Cross-domain Detection via Graph-induced Prototype Alignment [[CVPR2020 Oral]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Cross-Domain_Detection_via_Graph-Induced_Prototype_Alignment_CVPR_2020_paper.pdf) [[code]](https://github.com/ChrisAllenMing/GPA-detection)
- Multi-spectral Salient Object Detection by Adversarial Domain Adaptation [[Paper]](https://cse.sc.edu/~songwang/document/aaai20b.pdf)
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
- Cross-domain object detection using unsupervised image translation [[ESWA]](https://www.sciencedirect.com/science/article/abs/pii/S0957417421016328)
- Pixel and feature level based domain adaptation for object detection in autonomous driving [[Neurocomputing]](https://www.sciencedirect.com/science/article/pii/S092523121931149X?via%3Dihub)

**Arxiv**
- See Eye to Eye: A Lidar-Agnostic 3D Detection Framework for Unsupervised Multi-Target Domain Adaptation [[17 Nov 2021]](https://arxiv.org/abs/2111.09450)
- Unsupervised Domain Adaptive Object Detection using Forward-Backward Cyclic Adaptation [[3 Feb 2020]](https://arxiv.org/abs/2002.00575v1)
- Prior-based Domain Adaptive Object Detection for Adverse Weather Conditions [[29 Nov 2019]](https://arxiv.org/abs/1912.00070v1)
- Unsupervised Domain Adaptation for Object Detection via Cross-Domain Semi-Supervised Learning [[17 Nov 2019]](https://arxiv.org/abs/1911.07158v1)
- Curriculum Self-Paced Learning for Cross-Domain Object Detection [[15 Nov 2019]](https://arxiv.org/abs/1911.06849v1)
- SCL: Towards Accurate Domain Adaptive Object Detection via Gradient Detach Based Stacked Complementary Losses [[6 Nov 2019]](https://arxiv.org/abs/1911.02559v1)

### Semantic Segmentation

**Conference**

- PiPa: Pixel- and Patch-wise Self-supervised Learning for Domain Adaptative Semantic Segmentation [[ACM MM2023]](https://arxiv.org/abs/2211.07609) [[Pytorch]](https://github.com/chen742/PiPa)
- Refign: Align and Refine for Adaptation of Semantic Segmentation to Adverse Conditions [[WACV 2023]](https://arxiv.org/pdf/2207.06825.pdf) [[Pytorch]](https://github.com/brdav/refign)
- Deliberated Domain Bridging for Domain Adaptive Semantic Segmentation [[NeruIPS 2022]](https://arxiv.org/abs/2209.07695) [[Pytorch]](https://github.com/xiaoachen98/DDB)
- DecoupleNet: Decoupled Network for Domain Adaptive Semantic Segmentation [[ECCV 2022]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930362.pdf) [[Pytorch]](https://github.com/dvlab-research/DecoupleNet)
- HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation [[ECCV 2022]](https://arxiv.org/pdf/2204.13132) [[Pytorch]](https://github.com/lhoyer/HRDA)
- Style-Hallucinated Dual Consistency Learning for Domain Generalized Semantic Segmentation [[ECCV 2022]](https://arxiv.org/pdf/2204.02548.pdf) [[Pytorch]](https://github.com/HeliosZhao/SHADE)
- DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation [[CVPR 2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.pdf) [[Pytorch]](https://github.com/lhoyer/DAFormer)
- Plugging Self-Supervised Monocular Depth into Unsupervised Domain Adaptation for Semantic Segmentation [[WACV 2022]](https://openaccess.thecvf.com/content/WACV2022/papers/Cardace_Plugging_Self-Supervised_Monocular_Depth_Into_Unsupervised_Domain_Adaptation_for_Semantic_WACV_2022_paper.pdf)
- Shallow Features Guide Unsupervised Domain Adaptation for Semantic Segmentation at Class Boundaries [[WACV 2022]](https://openaccess.thecvf.com/content/WACV2022/papers/Cardace_Shallow_Features_Guide_Unsupervised_Domain_Adaptation_for_Semantic_Segmentation_at_WACV_2022_paper.pdf)
- Learning to Adapt via Latent Domains for Adaptive Semantic Segmentation [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/092cb13c22d51c22b9035a2b4fe76b00-Abstract.html)
- Dual Path Learning for Domain Adaptation of Semantic Segmentation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Cheng_Dual_Path_Learning_for_Domain_Adaptation_of_Semantic_Segmentation_ICCV_2021_paper.html)
- Exploring Robustness of Unsupervised Domain Adaptation in Semantic Segmentation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_Exploring_Robustness_of_Unsupervised_Domain_Adaptation_in_Semantic_Segmentation_ICCV_2021_paper.html)
- Multi-Anchor Active Domain Adaptation for Semantic Segmentation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Ning_Multi-Anchor_Active_Domain_Adaptation_for_Semantic_Segmentation_ICCV_2021_paper.html)
- LabOR: Labeling Only if Required for Domain Adaptive Semantic Segmentation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Shin_LabOR_Labeling_Only_if_Required_for_Domain_Adaptive_Semantic_Segmentation_ICCV_2021_paper.html)
- Self-Mutating Network for Domain Adaptive Segmentation in Aerial Images [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Lee_Self-Mutating_Network_for_Domain_Adaptive_Segmentation_in_Aerial_Images_ICCV_2021_paper.html)
- Geometric Unsupervised Domain Adaptation for Semantic Segmentation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Guizilini_Geometric_Unsupervised_Domain_Adaptation_for_Semantic_Segmentation_ICCV_2021_paper.html)
- Multi-Target Adversarial Frameworks for Domain Adaptation in Semantic Segmentation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Saporta_Multi-Target_Adversarial_Frameworks_for_Domain_Adaptation_in_Semantic_Segmentation_ICCV_2021_paper.html)
- BAPA-Net: Boundary Adaptation and Prototype Alignment for Cross-Domain Semantic Segmentation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_BAPA-Net_Boundary_Adaptation_and_Prototype_Alignment_for_Cross-Domain_Semantic_Segmentation_ICCV_2021_paper.html)
- BiMaL: Bijective Maximum Likelihood Approach to Domain Adaptation in Semantic Scene Segmentation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Truong_BiMaL_Bijective_Maximum_Likelihood_Approach_to_Domain_Adaptation_in_Semantic_ICCV_2021_paper.html)
- Uncertainty-Aware Pseudo Label Refinery for Domain Adaptive Semantic Segmentation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Uncertainty-Aware_Pseudo_Label_Refinery_for_Domain_Adaptive_Semantic_Segmentation_ICCV_2021_paper.html)
- Domain Adaptive Semantic Segmentation With Self-Supervised Depth Estimation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Domain_Adaptive_Semantic_Segmentation_With_Self-Supervised_Depth_Estimation_ICCV_2021_paper.html)
- Generalize Then Adapt: Source-Free Domain Adaptive Semantic Segmentation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Kundu_Generalize_Then_Adapt_Source-Free_Domain_Adaptive_Semantic_Segmentation_ICCV_2021_paper.pdf)
- DARCNN: Domain Adaptive Region-Based Convolutional Neural Network for Unsupervised Instance Segmentation in Biomedical Images [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Hsu_DARCNN_Domain_Adaptive_Region-Based_Convolutional_Neural_Network_for_Unsupervised_Instance_CVPR_2021_paper.pdf)
- DANNet: A One-Stage Domain Adaptation Network for Unsupervised Nighttime Semantic Segmentation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_DANNet_A_One-Stage_Domain_Adaptation_Network_for_Unsupervised_Nighttime_Semantic_CVPR_2021_paper.pdf)
- Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Prototypical_Pseudo_Label_Denoising_and_Target_Structure_Learning_for_Domain_CVPR_2021_paper.pdf)
- Complete & Label: A Domain Adaptation Approach to Semantic Segmentation of LiDAR Point Clouds [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yi_Complete__Label_A_Domain_Adaptation_Approach_to_Semantic_Segmentation_CVPR_2021_paper.pdf)
- Cluster, Split, Fuse, and Update: Meta-Learning for Open Compound Domain Adaptive Semantic Segmentation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Gong_Cluster_Split_Fuse_and_Update_Meta-Learning_for_Open_Compound_Domain_CVPR_2021_paper.pdf)
- PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Melas-Kyriazi_PixMatch_Unsupervised_Domain_Adaptation_via_Pixelwise_Consistency_Training_CVPR_2021_paper.pdf) [[Pytorch]](https://github.com/lukemelas/pixmatch)
- Learning to Relate Depth and Semantics for Unsupervised Domain Adaptation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Saha_Learning_To_Relate_Depth_and_Semantics_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.pdf) [[Pytorch]](https://github.com/susaha/ctrl-uda)
- Cross-View Regularization for Domain Adaptive Panoptic Segmentation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Cross-View_Regularization_for_Domain_Adaptive_Panoptic_Segmentation_CVPR_2021_paper.pdf)
- Semi-supervised Domain Adaptation based on Dual-level Domain Mixing for Semantic Segmentation [[CVPR2021]](https://arxiv.org/abs/2103.04705v1)
- MetaCorrection: Domain-aware Meta Loss Correction for Unsupervised Domain Adaptation in Semantic Segmentation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Guo_MetaCorrection_Domain-Aware_Meta_Loss_Correction_for_Unsupervised_Domain_Adaptation_in_CVPR_2021_paper.pdf)
- Coarse-to-Fine Domain Adaptive Semantic Segmentation with Photometric Alignment and Category-Center Regularization [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ma_Coarse-To-Fine_Domain_Adaptive_Semantic_Segmentation_With_Photometric_Alignment_and_Category-Center_CVPR_2021_paper.pdf) 
- Multi-Source Domain Adaptation with Collaborative Learning for Semantic Segmentation [[CVPR2021]](https://arxiv.org/abs/2103.04717v2)
- Source-Free Domain Adaptation for Semantic Segmentation [[CVPR2021]](https://arxiv.org/abs/2103.16372v1)
- Instance Adaptive Self-Training for Unsupervised Domain Adaptation [[ECCV 2020]](https://arxiv.org/abs/2008.12197) [[Pytorch]](https://github.com/bupt-ai-cz/IAST-ECCV2020)
- Cross-stained Segmentation from Renal Biopsy Images Using Multi-level Adversarial Learning [[ICASSP 2020]](https://arxiv.org/abs/2002.08587)
- Pixel-Level Cycle Association: A New Perspective for Domain Adaptive Semantic Segmentation [[NeurlIPS 2020]](https://arxiv.org/abs/2011.00147v1) [[Pytorch]](https://github.com/kgl-prml/Pixel-Level-Cycle-Association)
- Adversarial Style Mining for One-Shot Unsupervised Domain Adaptation [[NeurIPS2020]](https://proceedings.neurips.cc/paper/2020/hash/ed265bc903a5a097f61d3ec064d96d2e-Abstract.html) [[Pytorch]](https://github.com/RoyalVane/ASM)
- Semantically Adaptive Image-to-image Translation for Domain Adaptation of Semantic Segmentation [[BMVC2020]](https://arxiv.org/abs/2009.01166)
- Contextual-Relation Consistent Domain Adaptation for Semantic Segmentation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600698.pdf)
- Learning from Scale-Invariant Examples for Domain Adaptation in Semantic Segmentation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670290.pdf)
- Label-Driven Reconstruction for Domain Adaptation in Semantic Segmentation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720477.pdf)
- Unsupervised Domain Adaptation for Semantic Segmentation of NIR Images through Generative Latent Search [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510409.pdf)
- Domain Adaptive Semantic Segmentation Using Weak Labels [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540545.pdf)  
- Content-Consistent Matching for Domain Adaptive Semantic Segmentation [[ECCV2020]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590426.pdf) [[PyTorch]](https://github.com/Solacex/CCM)
- Cross-Domain Semantic Segmentation via Domain-Invariant Interactive Relation Transfer [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Lv_Cross-Domain_Semantic_Segmentation_via_Domain-Invariant_Interactive_Relation_Transfer_CVPR_2020_paper.pdf)
- Phase Consistent Ecological Domain Adaptation [[CVPR2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Phase_Consistent_Ecological_Domain_Adaptation_CVPR_2020_paper.pdf) [[Pytorch]](https://github.com/donglao/PCEDA)
- FDA: Fourier Domain Adaptation for Semantic Segmentation [[CVPR2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf) [[Pytorch]](https://github.com/YanchaoYang/FDA/blob/master/SStrain.py)
- Unsupervised Instance Segmentation in Microscopy Images via Panoptic Domain Adaptation and Task Re-weighting [[CVPR2020]](https://arxiv.org/abs/2005.02066v1)
- Unsupervised Intra-domain Adaptation for Semantic Segmentation through Self-Supervision [[CVPR2020 Oral]](https://arxiv.org/abs/2004.07703v1) [[Pytorch]](https://github.com/feipan664/IntraDA)
- Differential Treatment for Stuff and Things: A Simple Unsupervised Domain Adaptation Method for Semantic Segmentation [[CVPR2020]](https://arxiv.org/abs/2003.08040v1)
- Learning Texture Invariant Representation for Domain Adaptation of Semantic Segmentation [[CVPR2020]](https://arxiv.org/abs/2003.00867v2) [[Pytorch]](https://github.com/MyeongJin-Kim/Learning-Texture-Invariant-Representation)
- xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation [[CVPR2020]](https://arxiv.org/abs/1911.12676) [[Demo]](https://www.youtube.com/watch?v=WgvBBCEKQVE) [[code]](https://github.com/valeoai/xmuda)
- Unsupervised Scene Adaptation with Memory Regularization in vivo [[IJCAI2020]](https://arxiv.org/abs/1912.11164) [[code]](https://github.com/layumi/Seg-Uncertainty)
- Joint Adversarial Learning for Domain Adaptation in Semantic Segmentation [[AAAI2020]](https://aaai.org/ojs/index.php/AAAI/article/view/6169)
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
- CrDoCo: Pixel-level Domain Transfer with Cross-Domain Consistency [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_CrDoCo_Pixel-Level_Domain_Transfer_With_Cross-Domain_Consistency_CVPR_2019_paper.pdf) [[Project]](https://yunchunchen.github.io/CrDoCo/) [[Pytorch]](https://github.com/YunChunChen/CrDoCo-pytorch)
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
- SePiCo: Semantic-Guided Pixel Contrast for Domain Adaptive Semantic Segmentation [[TPAMI2023]](https://ieeexplore.ieee.org/document/10018569)[[arxiv version]](https://arxiv.org/abs/2204.08808)[[Pytorch]](https://github.com/BIT-DA/SePiCo)
- Adaptive Boosting for Domain Adaptation: Towards Robust Predictions in Scene Segmentation [[TIP2022]](https://arxiv.org/abs/2103.15685)[[Pytorch]](https://github.com/layumi/AdaBoost_Seg)
- Rectifying Pseudo Label Learning via Uncertainty Estimation for Domain Adaptive Semantic Segmentation [[IJCV2020]](https://arxiv.org/abs/2003.03773)[[Pytorch]](https://github.com/layumi/Seg-Uncertainty)
- Multi-level colonoscopy malignant tissue detection with adversarial CAC-UNet [[Neurocomputing 2021]](https://arxiv.org/abs/2006.15954) [[Pytorch]](https://github.com/bupt-ai-cz/CAC-UNet-DigestPath2019)
- Affinity Space Adaptation for Semantic Segmentation Across Domains [[TIP2020]](https://arxiv.org/abs/2009.12559)[[Pytorch]](https://github.com/idealwei/ASANet)
- Semantic-aware short path adversarial training for cross-domain semantic segmentation [[Neurocomputing 2019]](https://www.sciencedirect.com/science/article/pii/S0925231219315656#fig0002) 
- Weakly Supervised Adversarial Domain Adaptation for Semantic Segmentation in Urban Scenes [[TIP]](https://arxiv.org/abs/1904.09092v1)

**Arxiv**
- Class-Conditional Domain Adaptation on Semantic Segmentation [[27 Nov 2019]](https://arxiv.org/abs/1911.11981v1)
- Adversarial Learning and Self-Teaching Techniques for Domain Adaptation in Semantic Segmentation [[2 Sep 2019]](https://arxiv.org/abs/1909.00781v1)
- FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation [[8 Dec 2016]](https://arxiv.org/abs/1612.02649)
- BoMuDA: Boundless Multi-Source Domain Adaptive Segmentation in Unconstrained Environments [[13 Oct 2020]](https://arxiv.org/abs/2010.03523)[[Pytorch]](https://github.com/divyakraman/BoMuDA-Boundless-Multi-Source-Domain-Adaptive-Segmentation-in-Unstructured-Environments)
- SAfE: Self-Attention Based Unsupervised Road Safety Classification in Hazardous Environments [[27 Nov 2020]](https://arxiv.org/abs/2012.08939)[[Pytorch]](https://github.com/divyakraman/SAfE-Self-Attention-Based-Unsupervised-Road-Safety-Classification-in-Hazardous-Environments)
- Semantics-aware Multi-modal Domain Translation:From LiDAR Point Clouds to Panoramic Color Images [[26 Jun 2021]](https://arxiv.org/abs/2106.13974) [[Pytorch]](https://github.com/halmstad-University/TITAN-NET)

### Person Re-identification

**Conference**
- Learning to Generalize Unseen Domains via Memory-based Multi-Source Meta-Learning for Person Re-Identification [[CVPR 2021]](https://arxiv.org/pdf/2012.00417.pdf) [[Pytorch]](https://github.com/HeliosZhao/M3L)
- Group-aware Label Transfer for Domain Adaptive Person Re-identification [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Group-aware_Label_Transfer_for_Domain_Adaptive_Person_Re-identification_CVPR_2021_paper.pdf)
- Unsupervised Domain Adaptation in the Dissimilarity Space for Person Re-identification [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720154.pdf)
- Joint Visual and Temporal Consistency for Unsupervised Domain Adaptive Person Re-Identification [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690477.pdf)
- Joint Disentangling and Adaptation for Cross-Domain Person Re-Identification [[ECV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470086.pdf)
- Multiple Expert Brainstorming for Domain Adaptive Person Re-identification [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520579.pdf)
- Deep Credible Metric Learning for Unsupervised Domain Adaptation Person Re-identification [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530630.pdf)
- Unsupervised Domain Adaptation with Noise Resistible Mutual-Training for Person Re-identification [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560511.pdf)
- Generalizing Person Re-Identification by Camera-Aware Invariance Learning and Cross-Domain Mixup [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600222.pdf)
- AD-Cluster: Augmented Discriminative Clustering for Domain Adaptive Person Re-identification [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhai_AD-Cluster_Augmented_Discriminative_Clustering_for_Domain_Adaptive_Person_Re-Identification_CVPR_2020_paper.pdf)
- Smoothing Adversarial Domain Attack and P-Memory Reconsolidation for Cross-Domain Person Re-Identification [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Smoothing_Adversarial_Domain_Attack_and_P-Memory_Reconsolidation_for_Cross-Domain_Person_CVPR_2020_paper.pdf)
- Cross-Modal Cross-Domain Moment Alignment Network for Person Search [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jing_Cross-Modal_Cross-Domain_Moment_Alignment_Network_for_Person_Search_CVPR_2020_paper.pdf)
- Online Joint Multi-Metric Adaptation From Frequent Sharing-Subset Mining for Person Re-Identification [[CVPR2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Online_Joint_Multi-Metric_Adaptation_From_Frequent_Sharing-Subset_Mining_for_Person_CVPR_2020_paper.pdf)
- Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification [[ICLR2020]](https://openreview.net/forum?id=rJlnOhVYPS) [[Pytorch]](https://github.com/yxgeee/MMT)
- Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification [[ICCV2019 Oral]](https://arxiv.org/abs/1811.10144) [[Pytorch]](https://github.com/OasisYang/SSG)
- A Novel Unsupervised Camera-aware Domain Adaptation Framework for Person Re-identification [[ICCV2019]](https://arxiv.org/abs/1904.03425)
- Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification [[CVPR2019]](https://arxiv.org/abs/1904.01990v1) [[Pytorch]](https://github.com/zhunzhong07/ECN)
- Domain Adaptation through Synthesis for Unsupervised Person Re-identification [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Slawomir_Bak_Domain_Adaptation_through_ECCV_2018_paper.pdf)
- Person Transfer GAN to Bridge Domain Gap for Person Re-Identification [[CVPR2018]](https://arxiv.org/abs/1711.08565v2) 
- Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification [[CVPR2018]](https://arxiv.org/abs/1711.07027v3)

**Arxiv**
- Domain Adaptation for Semantic Segmentation via Patch-Wise Contrastive Learning [[arXiv 22 Apr 2021]](https://arxiv.org/abs/2104.11056)
- Structured Domain Adaptation for Unsupervised Person Re-identification [[arXiv 14 Mar 2020]](https://arxiv.org/abs/2003.06650)
- Domain Adaptive Attention Model for Unsupervised Cross-Domain Person Re-Identification [[arXiv 25 May 2019]](https://arxiv.org/abs/1905.10529)
- Camera Adversarial Transfer for Unsupervised Person Re-Identification [[arXiv 2 Apr 2019]](https://arxiv.org/abs/1904.01308)
- EANet: Enhancing Alignment for Cross-Domain Person Re-identification [[arXiv 29 Dec 2018]](https://arxiv.org/abs/1812.11369) [[Pytorch]](https://github.com/huanghoujing/EANet)
- One Shot Domain Adaptation for Person Re-Identification [[arXiv 26 Nov 2018]](https://arxiv.org/abs/1811.10144v1)
- Similarity-preserving Image-image Domain Adaptation for Person Re-identification [[arXiv 26 Nov 2018]](https://arxiv.org/abs/1811.10551v1)

### Sim-to-Real Transfer
**Conference**
- DIRL: Domain-Invariant Reperesentation Learning Approach for Sim-to-Real Transfer [[CoRL2020]](https://arxiv.org/abs/2011.07589) [[Project]](https://www.sites.google.com/view/dirl)

### Video Domain Adaptation

**Conference**
- Source-free Video Domain Adaptation by Learning Temporal Consistency for Action Recognition [[ECCV2022]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940144.pdf) [[Pytorch]](https://github.com/xuyu0010/ATCoN) [[Project]](https://xuyu0010.github.io/sfvda.html)
- Contrast and Mix: Temporal Contrastive Video Domain Adaptation with Background Mixing [[NeurIPS2021]](https://arxiv.org/pdf/2110.15128.pdf)
- Learning Cross-Modal Contrastive Features for Video Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_Learning_Cross-Modal_Contrastive_Features_for_Video_Domain_Adaptation_ICCV_2021_paper.html)
- Partial Video Domain Adaptation With Partial Adversarial Temporal Attentive Network [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_Partial_Video_Domain_Adaptation_With_Partial_Adversarial_Temporal_Attentive_Network_ICCV_2021_paper.html) [[Pytorch]](https://github.com/xuyu0010/PATAN)
- Domain Adaptive Video Segmentation via Temporal Consistency Regularization [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Guan_Domain_Adaptive_Video_Segmentation_via_Temporal_Consistency_Regularization_ICCV_2021_paper.html)
- Shuffle and Attend: Video Domain Adaptation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570664.pdf)
- Transferring Cross-Domain Knowledge for Video Sign Language Recognition [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Transferring_Cross-Domain_Knowledge_for_Video_Sign_Language_Recognition_CVPR_2020_paper.pdf)
- Action Segmentation with Joint Self-Supervised Temporal Domain Adaptation [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Action_Segmentation_With_Joint_Self-Supervised_Temporal_Domain_Adaptation_CVPR_2020_paper.pdf) [[Pytorch]](https://github.com/cmhungsteve/SSTDA)
- Transferring Cross-domain Knowledge for Video Sign Language Recognition [[CVPR2020 Oral]](https://arxiv.org/abs/2003.03703v2)
- Multi-Modal Domain Adaptation for Fine-Grained Action Recognition [[CVPR2020 Oral]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Munro_Multi-Modal_Domain_Adaptation_for_Fine-Grained_Action_Recognition_CVPR_2020_paper.pdf)
- Adversarial Cross-Domain Action Recognition with Co-Attention [[AAAI2020]](https://arxiv.org/abs/1912.10405v1)
- Generative Adversarial Networks for Video-to-Video Domain Adaptation [[Paper]](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-ChenJ.1453.pdf)
- Temporal Attentive Alignment for Large-Scale Video Domain Adaptation [[ICCV2019 Oral]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Temporal_Attentive_Alignment_for_Large-Scale_Video_Domain_Adaptation_ICCV_2019_paper.pdf) [[Pytorch]](https://github.com/olivesgatech/TA3N)
- Temporal Attentive Alignment for Video Domain Adaptation [[CVPRW 2019]](https://arxiv.org/abs/1905.10861v5) [[Pytorch]](https://github.com/olivesgatech/TA3N)


**Arxiv**
- Video Unsupervised Domain Adaptation with Deep Learning: A Comprehensive Survey [[17 Nov 2022]](https://arxiv.org/abs/2211.10412) [[project]](https://github.com/xuyu0010/awesome-video-domain-adaptation)
- Unsupervised Video Domain Adaptation: A Disentanglement Perspective [[15 Aug 2022]](https://arxiv.org/abs/2208.07365) [[Pyotrch]](https://github.com/ldkong1205/TranSVAE) [[Project]](https://ldkong.com/TranSVAE) [[Gradio Demo]](https://huggingface.co/spaces/ldkong/TranSVAE)
- Image to Video Domain Adaptation Using Web Supervision [[5 Aug 2019]](https://arxiv.org/abs/1908.01449)

### Medical Related
**Conference**
- PopGenAdapt: Semi-Supervised Domain Adaptation for Genotype-to-Phenotype Prediction in Underrepresented Populations [[PSB 2024]](https://psb.stanford.edu/psb-online/proceedings/psb24/comajoan.pdf)
- Cross-stained Segmentation from Renal Biopsy Images Using Multi-level Adversarial Learning [[ICASSP 2020]](https://arxiv.org/abs/2002.08587)
- What Can Be Transferred: Unsupervised Domain Adaptation for Endoscopic Lesions Segmentation [[Paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Dong_What_Can_Be_Transferred_Unsupervised_Domain_Adaptation_for_Endoscopic_Lesions_CVPR_2020_paper.pdf)
- Semantic-Transferable Weakly-Supervised Endoscopic Lesions Segmentation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Dong_Semantic-Transferable_Weakly-Supervised_Endoscopic_Lesions_Segmentation_ICCV_2019_paper.pdf)

**Journal**
- Multi-level colonoscopy malignant tissue detection with adversarial CAC-UNet [[Neurocomputing 2021]](https://arxiv.org/abs/2006.15954) [[Pytorch]](https://github.com/bupt-ai-cz/CAC-UNet-DigestPath2019)
  
**Arxiv**
- On-the-Fly Test-time Adaptation for Medical Image Segmentation [[10 Mar 2022]](https://arxiv.org/abs/2203.05574) [[Pytorch]](https://github.com/jeya-maria-jose/On-The-Fly-Adaptation)
- Target and task specific source-free domain adaptive image segmentation [[10 Mar 2023]](https://arxiv.org/abs/2203.15792) [[Pytorch]](https://github.com/Vibashan/tt-sfuda)
- Unsupervised Domain Adaptation via Disentangled Representations: Application to Cross-Modality Liver Segmentation [[arXiv 29 Aug 2019]](https://arxiv.org/abs/1907.13590)
- Synergistic Image and Feature Adaptation: Towards Cross-Modality Domain Adaptation for Medical Image Segmentation [[arXiv on 24 Jan 2019]](https://arxiv.org/abs/1901.08211v1)
- Unsupervised domain adaptation for medical imaging segmentation with self-ensembling [[arXiv 14 Nov 2018]](https://arxiv.org/abs/1811.06042v1)

### Monocular Depth Estimation
- Geometry-Aware Symmetric Domain Adaptation for Monocular Depth Estimation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Geometry-Aware_Symmetric_Domain_Adaptation_for_Monocular_Depth_Estimation_CVPR_2019_paper.pdf)
- Real-Time Monocular Depth Estimation using Synthetic Data with Domain Adaptation via Image Style Transfer [[CVPR2018]](http://breckon.eu/toby/publications/papers/abarghouei18monocular.pdf)

### 3D
**Conference**
- SPG: Unsupervised Domain Adaptation for 3D Object Detection via Semantic Point Generation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_SPG_Unsupervised_Domain_Adaptation_for_3D_Object_Detection_via_Semantic_ICCV_2021_paper.html)
- Sparse-to-Dense Feature Matching: Intra and Inter Domain Cross-Modal Learning in Domain Adaptation for 3D Semantic Segmentation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Peng_Sparse-to-Dense_Feature_Matching_Intra_and_Inter_Domain_Cross-Modal_Learning_in_ICCV_2021_paper.html)
- Unsupervised Domain Adaptive 3D Detection With Multi-Level Consistency [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Luo_Unsupervised_Domain_Adaptive_3D_Detection_With_Multi-Level_Consistency_ICCV_2021_paper.html)
- Domain-Adaptive Single-View 3D Reconstruction [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Pinheiro_Domain-Adaptive_Single-View_3D_Reconstruction_ICCV_2019_paper.pdf)

**Arxiv**
- MS3D++: Ensemble of Experts for Multi-Source Unsupervised Domain Adaptation in 3D Object Detection [[Arxiv 11 Aug 2023]](https://arxiv.org/abs/2308.05988) [[Pytorch]](https://github.com/darrenjkt/MS3D)

### Fine-Grained Domain
**Conference**
- Progressive Adversarial Networks for Fine-Grained Domain Adaptation [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Progressive_Adversarial_Networks_for_Fine-Grained_Domain_Adaptation_CVPR_2020_paper.pdf) [[Pytorch]](https://github.com/thuml/PAN)

### LiDAR

**Conference**
- SALUDA: Surface-based Automotive Lidar Unsupervised Domain Adaptation [[3DV 2024]](https://arxiv.org/abs/2304.03251) [[Pytorch]](https://github.com/valeoai/SALUDA)
- GIPSO: Geometrically Informed Propagation for Online Adaptation in 3D LiDAR Segmentation [[ECCV2022]](https://arxiv.org/abs/2207.09763)  [[Pytorch]](https://github.com/saltoricristiano/gipso-sfouda)
- CoSMix: Compositional Semantic Mix for Domain Adaptation in 3D LiDAR Segmentation [[ECCV2022]](https://arxiv.org/abs/2207.09778) [[Pytorch]]( https://github.com/saltoricristiano/cosmix-uda)


**ArXiv**
- ConDA: Unsupervised Domain Adaptation for LiDAR Segmentation via Regularized Domain Concatenation [[13 Mar 2022]](https://arxiv.org/abs/2111.15242)

### Remote Sensing

**Journal**
- Open-Set Black-Box Domain Adaptation for Remote Sensing Image Scene Classification [[GRSL 2023]](https://ieeexplore.ieee.org/abstract/document/10210386)


### Others
**Conference**
- RefRec: Pseudo-labels Refinement via Shape Reconstruction for Unsupervised 3D Domain Adaptation [[3DV 2021 Oral]](https://ieeexplore.ieee.org/abstract/document/9665900)
- Unsupervised Domain Adaptation in LiDAR Semantic Segmentation with Self-Supervision and Gated Adapters [[ICRA2022]](https://arxiv.org/abs/2107.09783)
- RDA: Robust Domain Adaptation via Fourier Adversarial Attacking [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_RDA_Robust_Domain_Adaptation_via_Fourier_Adversarial_Attacking_ICCV_2021_paper.html)
- Geometry-Aware Self-Training for Unsupervised Domain Adaptation on Object Point Clouds [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Zou_Geometry-Aware_Self-Training_for_Unsupervised_Domain_Adaptation_on_Object_Point_Clouds_ICCV_2021_paper.html)
- Tune It the Right Way: Unsupervised Validation of Domain Adaptation via Soft Neighborhood Density [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Saito_Tune_It_the_Right_Way_Unsupervised_Validation_of_Domain_Adaptation_ICCV_2021_paper.html)
- PIT: Position-Invariant Transform for Cross-FoV Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Gu_PIT_Position-Invariant_Transform_for_Cross-FoV_Domain_Adaptation_ICCV_2021_paper.html)
- Self-Supervised Domain Adaptation for Forgery Localization of JPEG Compressed Images [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Rao_Self-Supervised_Domain_Adaptation_for_Forgery_Localization_of_JPEG_Compressed_Images_ICCV_2021_paper.html)
- Unsupervised Real-World Super-Resolution: A Domain Adaptation Perspective [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Unsupervised_Real-World_Super-Resolution_A_Domain_Adaptation_Perspective_ICCV_2021_paper.html)
- Adversarial Robustness for Unsupervised Domain Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Awais_Adversarial_Robustness_for_Unsupervised_Domain_Adaptation_ICCV_2021_paper.html)
- Collaborative Optimization and Aggregation for Decentralized Domain Generalization and Adaptation [[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_Collaborative_Optimization_and_Aggregation_for_Decentralized_Domain_Generalization_and_Adaptation_ICCV_2021_paper.html)
- Adaptive Cross-Modal Prototypes for Cross-Domain Visual-Language Retrieval [[CVPR2021]](https://openaccess.thecvf.com/CVPR2021?day=all)
- Spatio-temporal Contrastive Domain Adaptation for Action Recognition [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Song_Spatio-temporal_Contrastive_Domain_Adaptation_for_Action_Recognition_CVPR_2021_paper.pdf)
- Regressive Domain Adaptation for Unsupervised Keypoint Detection [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Jiang_Regressive_Domain_Adaptation_for_Unsupervised_Keypoint_Detection_CVPR_2021_paper.pdf)
- From Synthetic to Real: Unsupervised Domain Adaptation for Animal Pose Estimation [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_From_Synthetic_to_Real_Unsupervised_Domain_Adaptation_for_Animal_Pose_CVPR_2021_paper.pdf) [[code coming soon]](https://github.com/chaneyddtt/UDA-Animal-Pose)
- Meta Self-Learning for Multi-Source Domain Adaptation: A Benchmark [[ICCV Workshop 2021]](https://arxiv.org/abs/2108.10840) [[Pytorch]](https://github.com/bupt-ai-cz/Meta-SelfLearning)
- Adapting Neural Architectures Between Domains [[NeurlPS 2020]](https://github.com/liyxi/AdaptNAS)
- Unsupervised Domain Attention Adaptation Network for Caricature Attribute Recognition [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530018.pdf)
- A Broader Study of Cross-Domain Few-Shot Learning [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720120.pdf)
- Label-Noise Robust Domain Adaptation [[ICML2020]](https://proceedings.icml.cc/static/paper_files/icml/2020/1942-Paper.pdf)
- Unsupervised Domain Adaptation of a Pretrained Cross-Lingual Language Model [[IJCAI2020]](https://www.ijcai.org/Proceedings/2020/0508.pdf)
- Domain Adaptation for Semantic Parsing [[IJCAI2020]](https://www.ijcai.org/Proceedings/2020/0515.pdf)
- Bridging Cross-Tasks Gap for Cognitive Assessment via Fine-Grained Domain Adaptation [[IJCAI2020]](https://www.ijcai.org/Proceedings/2020/0597.pdf)
- Clarinet: A One-step Approach Towards Budget-friendly Unsupervised Domain Adaptation [[IJCAI2020]](https://www.ijcai.org/Proceedings/2020/0350.pdf)
- Weakly-Supervised Domain Adaptation via GAN and Mesh Model for Estimating 3D Hand Poses Interacting Objects [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Baek_Weakly-Supervised_Domain_Adaptation_via_GAN_and_Mesh_Model_for_Estimating_CVPR_2020_paper.pdf)
- One-Shot Domain Adaptation for Face Generation [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_One-Shot_Domain_Adaptation_for_Face_Generation_CVPR_2020_paper.pdf)
- Learning Meta Face Recognition in Unseen Domains [[CVPR2020 Oral]](hhttp://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Learning_Meta_Face_Recognition_in_Unseen_Domains_CVPR_2020_paper.pdf) [[code]](https://github.com/cleardusk/MFR)
- Cross-Domain Document Object Detection: Benchmark Suite and Method [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Cross-Domain_Document_Object_Detection_Benchmark_Suite_and_Method_CVPR_2020_paper.pdf) [[code]](https://github.com/kailigo/cddod)
- StereoGAN: Bridging Synthetic-to-Real Domain Gap by Joint Optimization of Domain Translation and Stereo Matching [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_StereoGAN_Bridging_Synthetic-to-Real_Domain_Gap_by_Joint_Optimization_of_Domain_CVPR_2020_paper.pdf)
- Domain Adaptation for Image Dehazing [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Shao_Domain_Adaptation_for_Image_Dehazing_CVPR_2020_paper.pdf)
- Probability Weighted Compact Feature for Domain Adaptive Retrieval [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_Probability_Weighted_Compact_Feature_for_Domain_Adaptive_Retrieval_CVPR_2020_paper.pdf) [[code]](https://github.com/fuxianghuang1/PWCF)
- Disparity-Aware Domain Adaptation in Stereo Image Restoration [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yan_Disparity-Aware_Domain_Adaptation_in_Stereo_Image_Restoration_CVPR_2020_paper.pdf)
- Multi-Path Learning for Object Pose Estimation Across Domains [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Sundermeyer_Multi-Path_Learning_for_Object_Pose_Estimation_Across_Domains_CVPR_2020_paper.pdf)
- Unsupervised Domain Adaptation for 3D Human Pose Estimation [[ACM MM2019]](https://dl.acm.org/citation.cfm?id=3351052)
- PointDAN: A Multi-Scale 3D Domain Adaption Network for Point Cloud Representation [[NeurIPS 2019]](https://arxiv.org/abs/1911.02744v1) [[code]](https://github.com/canqin001/PointDAN)
- Deep Head Pose Estimation Using Synthetic Images and Partial Adversarial Domain Adaption for Continuous Label Spaces [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Kuhnke_Deep_Head_Pose_Estimation_Using_Synthetic_Images_and_Partial_Adversarial_ICCV_2019_paper.pdf)
- Cross-Domain Adaptation for Animal Pose Estimation [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cao_Cross-Domain_Adaptation_for_Animal_Pose_Estimation_ICCV_2019_paper.pdf)
- GA-DAN: Geometry-Aware Domain Adaptation Network for Scene Text Detection and Recognition [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhan_GA-DAN_Geometry-Aware_Domain_Adaptation_Network_for_Scene_Text_Detection_and_ICCV_2019_paper.pdf)
- Accelerating Deep Unsupervised Domain Adaptation with Transfer Channel Pruning [[IJCNN]](https://arxiv.org/abs/1904.02654)
- Adversarial Adaptation of Scene Graph Models for Understanding Civic Issues [[WWW2019]](https://arxiv.org/abs/1901.10124)
- Cross-Dataset Adaptation for Visual Question Answering [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chao_Cross-Dataset_Adaptation_for_CVPR_2018_paper.pdf)
- Cross-domain fault diagnosis through optimal transport for a CSTR process [[DYCOPS2022]](https://www.sciencedirect.com/science/article/pii/S2405896322009727) [[Code]](https://github.com/eddardd/CrossDomainFaultDiagnosis)

**Journal**
- DASGIL: Domain Adaptation for Semantic and Geometric-Aware Image-Based Localization [[TIP2020]](https://ieeexplore.ieee.org/document/9296559) [[Pytorch]](https://github.com/HanjiangHu/DASGIL) 
- An Unsupervised Domain Adaptation Scheme for Single-Stage Artwork Recognition in Cultural Sites [[Image and Vision Computing 2020]](https://arxiv.org/abs/2008.01882v3) [[Pytorch]](https://github.com/fpv-iplab/DA-RetinaNet) [[Project]](https://iplab.dmi.unict.it/EGO-CH-OBJ-UDA/)
- Multi-source transfer learning of time series in cyclical manufacturing [[JIntellManuf2020]](https://link.springer.com/article/10.1007/s10845-019-01499-4)
- Domain adaptation for regression under Beer-Lambert's law [[KBS2020]](https://www.sciencedirect.com/science/article/abs/pii/S0950705120305761)


**Arxiv**
- Open-Ended Visual Question Answering by Multi-Modal Domain Adaptation [[11 Nov 2019]](https://arxiv.org/abs/1911.04058)
- DANE: Domain Adaptive Network Embedding [[arXiv 3 Jun 2019]](https://arxiv.org/abs/1906.00684v1)
- Active Adversarial Domain Adaptation [[arXiv 16 Apr 2019]](https://arxiv.org/abs/1904.07848v1)

## Benchmarks
- Kurcuma: a kitchen utensil recognition collection for unsupervised domain adaptation [[PAA2023]](https://link.springer.com/article/10.1007/s10044-023-01147-x) [[Dataset]](https://www.dlsi.ua.es/~jgallego/datasets/kurcuma/)
- Meta Self-Learning for Multi-Source Domain Adaptation: A Benchmark [[ICCV Workshop 2021]](https://arxiv.org/abs/2108.10840) [[Pytorch]](https://github.com/bupt-ai-cz/Meta-SelfLearning)
- LLVIP: A Visible-infrared Paired Dataset for Low-light Vision [[ICCV Workshop 2021]](https://arxiv.org/abs/2108.10831) [[Pytorch]](https://github.com/bupt-ai-cz/LLVIP)
- Syn2Real: A New Benchmark forSynthetic-to-Real Visual Domain Adaptation [[arXiv 26 Jun]](https://arxiv.org/abs/1806.09755v1) [[Project]](http://ai.bu.edu/syn2real/)
- Benchmarking Neural Network Robustness to Common Corruptions and Perturbations (ImageNet-C) [[ICLR 2019]](https://arxiv.org/abs/1903.12261) [[PyTorch]](https://github.com/hendrycks/robustness)

# Library
- [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)
- [deep-transfer-learning: a PyTorch library for deep transfer learning](https://github.com/easezyc/deep-transfer-learning)
- [salad: a Semi-supervised Adaptive Learning Across Domains](https://domainadaptation.org/)
- [Dassl: a PyTorch toolbox for domain adaptation and semi-supervised learning](https://github.com/KaiyangZhou/Dassl.pytorch)
- [joliGEN: an integrated framework for training custom generative AI image-to-image models](https://github.com/jolibrain/joliGEN)

# Lectures and Tutorials
- A Primer on Domain Adaptation [[PDF]](https://arxiv.org/abs/2001.09994v2)


# Other Resources
- [transferlearning](https://github.com/jindongwang/transferlearning)
