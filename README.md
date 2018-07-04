# awsome-domain-adaptation

This repo is a collection of AWESOME things about domian adaptation,including papers,code etc.Feel free to star and fork.

## Contents
- [Overview](#overview)
- [Theory](#theory)
- [Adversarial DA](#adversarial-da)
- [Network DA](#network-da)
- [Image-to-Image Translation](#image-to-image-translation)
- [Other Methods](#other-methods)
- [Open Set DA](#open-set-da)
- [Multi-source DA](#multi-source-da)
- [Applications](#applications)

## Overview
- Deep Visual Domain Adaptation: A Survey [[arXiv 2018]](https://arxiv.org/abs/1802.03601v4)
- Domain Adaptation for Visual Applications: A Comprehensive Survey  [[arXiv 2017]](https://arxiv.org/abs/1702.05374)

## Theory
- Analysis of Representations for Domain Adaptation [[NIPS2006]](https://papers.nips.cc/paper/2983-analysis-of-representations-for-domain-adaptation)
- A theory of learning from different domains [[ML2010]](https://link.springer.com/content/pdf/10.1007%2Fs10994-009-5152-4.pdf)
- Learning Bounds for Domain Adaptation [[NIPS2007]](http://papers.nips.cc/paper/3212-learning-bounds-for-domain-adaptation)
 
## Adversarial DA
- Augmented Cyclic Adversarial Learning for Domain Adaptation [[arXiv 1 Jul 2018]](https://arxiv.org/abs/1807.00374v1)
- Unsupervised Domain Adaptation with Adversarial Residual Transform Networks [[arXiv 25 Apr 2018]](https://arxiv.org/abs/1804.09578)
- Simple Domain Adaptation with Class Prediction Uncertainty Alignment [[arXiv  12 Apr 2018]](https://arxiv.org/abs/1804.04448)
- Causal Generative Domain Adaptation Networks [[Arxiv 12 Apr 2018]](https://arxiv.org/abs/1804.04333)
- Detach and Adapt: Learning Cross-Domain Disentangled Deep Representation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Detach_and_Adapt_CVPR_2018_paper.pdf)
- Importance Weighted Adversarial Nets for Partial Domain Adaptation [[CVPR2018]](https://arxiv.org/abs/1803.09210)
- Maximum Classifier Discrepancy for Unsupervised Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf) [[Pytorch(Official)]](https://github.com/mil-tokyo/MCD_DA)
- Domain Generalization with Adversarial Feature Learning [[CVPR2018]](https://www.researchgate.net/profile/Haoliang_Li5/publication/324691022_Domain_Generalization_with_Adversarial_Feature_Learning/links/5add5329458515c60f5f3187/Domain-Generalization-with-Adversarial-Feature-Learning.pdf)
- Adversarial Feature Augmentation for Unsupervised Domain Adaptation [[CVPR2018]](https://arxiv.org/abs/1711.08561) [[TensorFlow(Official)]](https://github.com/ricvolpi/adversarial-feature-augmentation)
- Partial Transfer Learning with Selective Adversarial Networks [[CVPR2018]](https://arxiv.org/abs/1707.07901)[[paper weekly]](http://www.paperweekly.site/papers/1388)
- Duplex Generative Adversarial Network for Unsupervised Domain Adaptation [[CVPR2018]](http://vipl.ict.ac.cn/uploadfile/upload/2018041610083083.pdf) [[Pytorch(Official)]](http://vipl.ict.ac.cn/view_database.php?id=6)
- Generate To Adapt: Aligning Domains using Generative Adversarial Networks [[CVPR2018]](https://arxiv.org/abs/1704.01705) [[Pytorch(Official)]](https://github.com/yogeshbalaji/Generate_To_Adapt)
- Image to Image Translation for Domain Adaptation [[CVPR2018]](https://arxiv.org/abs/1712.00479)
- Unsupervised Domain Adaptation with Similarity Learning [[CVPR2018]](https://arxiv.org/abs/1711.08995)
- Conditional Generative Adversarial Network for Structured Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hong_Conditional_Generative_Adversarial_CVPR_2018_paper.pdf)
- Collaborative and Adversarial Network for Unsupervised Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Collaborative_and_Adversarial_CVPR_2018_paper.pdf)
- Re-Weighted Adversarial Adaptation Network for Unsupervised Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Re-Weighted_Adversarial_Adaptation_CVPR_2018_paper.pdf)
- Multi-Adversarial Domain Adaptation [[AAAI2018]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/multi-adversarial-domain-adaptation-aaai18.pdf) [[Caffe(Official)]](https://github.com/thuml/MADA)
- Wasserstein Distance Guided Representation Learning for Domain Adaptation [[AAAI2018]](https://arxiv.org/abs/1707.01217) [[TensorFlow(official)]](https://github.com/RockySJ/WDGRL)
- Incremental Adversarial Domain Adaptation for Continually Changing Environments [[ICRA2018]](https://arxiv.org/abs/1712.07436)
- A DIRT-T Approach to Unsupervised Domain Adaptation [[ICLR2018 Poster]](https://openreview.net/forum?id=H1q-TM-AW) [[Tensorflow(Official)]](https://github.com/RuiShu/dirt-t)
- Label Efficient Learning of Transferable Representations acrosss Domains and Tasks [[NIPS2017]](http://vision.stanford.edu/pdf/luo2017nips.pdf) [[Project]](http://alan.vision/nips17_website/)
- Addressing Appearance Change in Outdoor Robotics with Adversarial Domain Adaptation [[IROS2017]](https://arxiv.org/abs/1703.01461)
- Adversarial Discriminative Domain Adaptation [[CVPR2017]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf)  [[Tensorflow(Official)]](https://github.com/erictzeng/adda) [[Pytorch]](https://github.com/corenel/pytorch-adda)
- Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks [[CVPR2017]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bousmalis_Unsupervised_Pixel-Level_Domain_CVPR_2017_paper.pdf) [[Tensorflow(Official)]](https://github.com/tensorflow/models/tree/master/research/domain_adaptation) [[Pytorch]](https://github.com/vaibhavnaagar/pixelDA_GAN)
- Domain Separation Networks [[NIPS2016]](http://papers.nips.cc/paper/6254-domain-separation-networks)
- Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation [[ECCV2016]](https://arxiv.org/abs/1607.03516)
- Domain-Adversarial Training of Neural Networks [[JMLR2016]](http://www.jmlr.org/papers/volume17/15-239/15-239.pdf)
- Unsupervised Domain Adaptation by Backpropagation [[ICML2015]](http://proceedings.mlr.press/v37/ganin15.pdf) [[Caffe(Official)]](https://github.com/ddtm/caffe/tree/grl) [[Tensorflow]](https://github.com/shucunt/domain_adaptation) [[Pytorch]](https://github.com/fungtion/DANN)

## Incremental DA
- Incremental Adversarial Domain Adaptation for Continually Changing Environments [[ICRA2018]](https://arxiv.org/abs/1712.07436)
- Continuous Manifold based Adaptation for Evolving Visual Domains [[CVPR2014]](https://people.eecs.berkeley.edu/~jhoffman/papers/Hoffman_CVPR2014.pdf)

## Network DA
- Boosting Domain Adaptation by Discovering Latent Domains [[CVPR2018]](https://arxiv.org/abs/1805.01386)
- Residual Parameter Transfer for Deep Domain Adaptation [[CVPR2018]](https://arxiv.org/abs/1711.07714)
- Deep Asymmetric Transfer Network for Unbalanced Domain Adaptation [[AAAI2018]](http://media.cs.tsinghua.edu.cn/~multimedia/cuipeng/papers/DATN.pdf)
- Deep CORAL: Correlation Alignment for Deep Domain Adaptation [[ECCV2016]](https://arxiv.org/abs/1607.01719)
- Deep Domain Confusion: Maximizing for Domain Invariance [[Arxiv 2014]](https://arxiv.org/abs/1412.3474)

## Image-to-Image Translation
- Multimodal Unsupervised Image-to-Image Translation [[arXiv]](https://arxiv.org/abs/1804.04732) [[Pytorch(Official)]](https://github.com/nvlabs/MUNIT)
- StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation [[CVPR2018]](https://arxiv.org/abs/1711.09020) [[Pytorch(Official)]](https://github.com/yunjey/StarGAN)
- Conditional Image-to-Image Translation [[CVPR2018]](https://arxiv.org/abs/1805.00251v1)
- Toward Multimodal Image-to-Image Translation [[NIPS2017]](https://arxiv.org/abs/1711.11586) [[Project]](https://junyanz.github.io/BicycleGAN/) [[Pyotorch(Official)]](https://github.com/junyanz/BicycleGAN)
- Unsupervised Image-to-Image Translation Networks [[NIPS2017]](http://papers.nips.cc/paper/6672-unsupervised-image-to-image-translation-networks) [[Pytorch(Official)]](https://github.com/mingyuliutw/unit)
- Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks [[ICCV2017(extended version)]](https://arxiv.org/abs/1703.10593v4) [[Pytorch(Official)]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- Image-to-Image Translation with Conditional Adversarial Nets [[CVPR2017]](https://arxiv.org/abs/1611.07004)  [[Project]](https://phillipi.github.io/pix2pix/) [[Pytorch(Official)]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 
- Learning to Discover Cross-Domain Relations with Generative Adversarial Networks [[ICML2017]](https://arxiv.org/abs/1703.05192) [[Pytorch(Official)]](https://github.com/SKTBrain/DiscoGAN)
- Unsupervised Cross-Domain Image Generation [[ICLR2017 Poster]](https://openreview.net/forum?id=Sk2Im59ex) [[TensorFlow]](https://github.com/yunjey/domain-transfer-network)
- Coupled Generative Adversarial Networks [[NIPS2016]](http://papers.nips.cc/paper/6544-coupled-generative-adversarial-networks) [[Poytorch(Official)]](https://github.com/mingyuliutw/cogan)

## Other Methods
- Unsupervised Domain Adaptation with Distribution Matching Machines [[AAAI2018]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/distribution-matching-machines-aaai18.pdf)
- Self-Ensembling for Visual Domain Adaptation [[ICLR2018 Poster]](https://openreview.net/forum?id=rkpoTaxA-)
- Minimal-Entropy Correlation Alignment for Unsupervised Deep Domain Adaptation [[ICLR2018 Poster]](https://openreview.net/forum?id=rJWechg0Z)
- Aligning Infinite-Dimensional Covariance Matrices in Reproducing Kernel Hilbert Spaces for Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Aligning_Infinite-Dimensional_Covariance_CVPR_2018_paper.pdf)
- Learning Transferrable Representations for Unsupervised Domain Adaptation [[NIPS2016]](http://papers.nips.cc/paper/6360-learning-transferrable-representations-for-unsupervised-domain-adaptation)

## Open Set DA
- Open Set Domain Adaptation by Backpropagation [[arXiv 27 Apr 2018]](https://arxiv.org/abs/1804.10427)
- Open Set Domain Adaptation [[ICCV2017]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Busto_Open_Set_Domain_ICCV_2017_paper.pdf)

## Multi source DA
- Deep Cocktail Network: Multi-source Unsupervised Domain Adaptation with Category Shift [[CVPR2018]](https://arxiv.org/abs/1803.00830)

## Applications
- Cross-Domain Weakly-Supervised Object Detection Through Progressive Domain Adaptation [[CVPR2018]](https://arxiv.org/abs/1803.11365)
- Real-Time Monocular Depth Estimation using Synthetic Data with Domain Adaptation via Image Style Transfer [[CVPR2018]](http://breckon.eu/toby/publications/papers/abarghouei18monocular.pdf)
- Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification [[CVPR2018]](https://arxiv.org/abs/1711.07027v3)
- Learning From Synthetic Data: Addressing Domain Shift for Semantic Segmentation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sankaranarayanan_Learning_From_Synthetic_CVPR_2018_paper.pdf)
- Domain Adaptive Faster R-CNN for Object Detection in the Wild [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Domain_Adaptive_Faster_CVPR_2018_paper.pdf)
- Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes [[ICCV2017]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Curriculum_Domain_Adaptation_ICCV_2017_paper.pdf)
