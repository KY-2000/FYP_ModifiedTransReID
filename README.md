# Modified_TransReID

## Introduction
Person re-identification(Person ReID) is one of the hot research topic in computer
vision. The main research purpose is to accurately identify the same person across multiple
non-overlapping surveillance cameras with different viewpoints. The research is significant
because it has wide real world application such as video surveillance, security detection, and crowd management. 

Person ReID has encountered various challenges such as occlusions, illumination
changes, viewpoint changes and complex background. Currently, feature representation
learning and metric learning are used to determine features from the person images. Researchers also studied and applied deep learning-based methods to address these
challenges. Convolutional neural networks(CNNs) are been used as backbone networks to
extract features from images. 


## ViT Limitation
However, there are a few limitations from CNNs. CNNs mainly focus on small
discriminative regions. Besides, since CNNs use convolution and downsampling, the
information of images is not retained completely. Thus, the usage of Vision
Transformer(ViT) as backbone network is becoming increasingly popular to address these
challenges. ViTs have stronger sequence information capturing ability due to the usage of
self attention. Despite of these advantages compared to CNNs, ViTs still suffer from high
computational complexity and low efficiency. Researchers have proposed various ViT
variants to counter this issue and adapt to perosn ReID, such as AAformer, SKIPAT, OH-Former and Per-former. 

In addition, as global and local features alone cannot fully differentiate the differences
between two persons, person ReID researchers has gradually shifted the direction towards
the integration of auxiliary information. However, collecting and organizing auxiliary
information is complex and increases the complexity of the network. TransReID and PFT
provides different perspective to address the challenges without complex auxiliary
information and related network. Therefore, relying on existing feature maps for more
detailed data augmentation and feature fusion, as well as better utilizing existing data and
feature information, is also an important research direction.

## Baseline
In this paper, TransReID is used as baseline, which is the first pure ViT approach to
solve person ReID challenges. TransReID do have some innovation to enhance the feature
extraction ability. First of all, TransReID uses overlapping patches to retain local adjacent
structure. Side Information Embedding(SIE) encodes camera ID and viewpoint ID
information, incorporates non-visual information in patch sequence. Jigsaw Patch
Module(JPM) is proposed to rearrange patch sequence by shift and shuffle operations, expands long-range dependencies and extracts more discriminative, robust features. 
TransReID Github Page: https://github.com/damo-cv/TransReID

## Methodology
To address these issues, this paper proposes a person ReID network based on
TransReID. This paper improves the efficiency and performance based on data
augmentation and feature fusion & reconstruction, which includes four modules. Among
them, the data augmentation-based person ReID network contains modules below, which
are Patch Full-Dimension Enhancement (PFDE) module, Patch Dropout module and Batch
DropBlock(BDB) module. 

### (1) PFDE Module
PFDE module is a simple yet effective module to enhance feature representation. Besides, this module is easy to deploy to any ViT. First, the module takes patch sequence
without [cls] token, positional embeddings and side information embeddings as input. The
size of patch sequence is where N is the number of patches and D is the dimension of each
patch. A learnable tensor with same size as the patch sequence is initialized with all
elements equal to 1. The Hadamard Product between patch sequence and learnable tensor
is calculated to obtain enhanced patch sequence.

The element value of the learnable tensor will be decreased if there are less
information or noises in the patch, thus the impact of these patches to the network will be
decreased. While the element value of the learnable tensor will be increased if there are
more information in the patch, the impact of these patches to the network will be increased. This module successfully improves performance without adding more training time, computational and memory cost. 

### (2) Patch Dropout Module
Patch Dropout module is used to increase the efficiency of ViTs without changing to
other ViT variations, thus successfully avoids high workload to change the backbone
networks. Since not all patches are equally important, the number of patch in patch
sequence can be decreased, so that the input is decreased, thus improves model efficiency.

Patch Dropout module takes patch sequence, including [cls] token, positional
embeddings and other auxiliary information as input. [cls] token will be retained by default. The module randomly retains certain amount of patch, according to the patch keep rate. According to the experiment result, this module has significantly decreased the
computational and memory cost, while only caused minor performance degradation.

### (3) BDB Module [WIP]
BDB module is designed to enhance the network ability to extract the attentive feature
of local regions . However, BDB module is only available to CNNs. Therefore, we propose
BDB module that can be applied to ViT.[cls] token will not be masked so that the global feature extracted by backbone network is
retained. After the mask is formed, the Hadamard Product between patch sequence and
mask is calculated to obtain the enhanced patch sequence. 

### (4) FRM
At the same time, the feature fusion and reconstruction-based person ReID network
contains module below, which is Fusion & Reconstruction Module(FRM). FRM module is
used to enhance the spatial correlation of the output feature block, reduce noise impact and
improves model performance by enhancing the impact of more significant patch sequence. 

There are more important regions and less important regions within an image. According to the calculation result of cosine similarity between local features blocks and
global feature, more important regions within an image are mostly located in the middle of
patch sequence. Besides, the front and end of the patch sequence usually have noises from
background or other items, thus may cause disturbance to the model. However, it doesn’t
mean that we should get rid of these less important regions since there are still some
information in these regions that might can be used to determine different person.

First, FRM splits the [cls] token and patch sequence so that [cls] token is not affected. The patch sequence will be split into four different sequences, which are head sequence, second sequence, third sequence and tail sequence. To decrease the impact of less
important regions to the output, head sequence and tail sequence will firstly multiplied by
feature fusion scale,which is smaller than 1. Head sequence and tail sequence will
fuse with second patch sequence and third patch sequence respectively. These sequences
and [cls] token will be concatenated to form new patch sequence.
