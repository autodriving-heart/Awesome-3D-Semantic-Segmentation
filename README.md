## Awesome-3D-Semantic-Segmentation

本仓库由[公众号【自动驾驶之心】](https://mp.weixin.qq.com/s?__biz=Mzg2NzUxNTU1OA==&mid=2247542481&idx=1&sn=c6d8609491a128233c3c3b91d68d22a6&chksm=ceb80b18f9cf820e789efd75947633aec9d2f1e8b58c29e5051c05a64b21ae63c244d54886a1&token=11182364&lang=zh_CN#rd) 团队整理，欢迎关注，一览最前沿的技术分享！

自动驾驶之心是国内首个自动驾驶开发者社区！这里有最全面有效的自动驾驶与AI学习路线（感知/定位/融合）和自动驾驶与AI公司内推机会！


[[Awesome-3D-Semantic-Segmentation]](https://github.com/TianhaoFu/Awesome-3D-Semantic-Segmentation)

[[Awesome-3D-Point-Cloud-Semantic-Segement]](https://github.com/lizhangjie316/Awesome-3D-Point-Cloud-Semantic-Segement)



## 一、Overview

### 1.**3D segmentation overview, a comprehensive investigation of 3D semantics/instance/part segmentation, and future research directions 

Deep Learning based 3D Segmentation: A Survey

## 二、3D Semantic Segmentation

### 1. RGB-D based

3D Graph Neural Networks for RGBD Semantic Segmentation

Cascaded Feature Network for Semantic Segmentation of RGB-D Images

Collaborative Deconvolutional Neural Networks for Joint Depth Estimation and Semantic Segmentation

Depth-aware CNN for RGB-D Segmentation

Exploiting Depth from Single Monocular Images for Object Detection and Semantic Segmentation

FuseNet: Incorporating Depth into Semantic Segmentation via Fusion-based CNN Architecture

Incorporating Depth into both CNN and CRF for Indoor Semantic Segmentation

Joint Semantic Segmentation and Depth Estimation with Deep Convolutional Networks

Learning Common and Specic Features for RGB-D Semantic Segmentation with Deconvolutional Networks

Learning Rich Features from RGB-D Images for Object Detection and Segmentation

Locality-Sensitive Deconvolution Networks with Gated Fusion for RGB-D Indoor Semantic Segmentation

LSTM-CF: Unifying Context Modeling and Fusion with LSTMs for RGB-D Scene Labeling

RGB-D joint modelling with scene geometric information for indoor semantic segmentation

RGB-D Scene Labeling with Multimodal Recurrent Neural Networks

Semantic segmentation of RGBD images based on deep depth regression

### Projected image based

Deep Projective 3D Semantic Segmentation

PointSeg: Real-Time Semantic Segmentation Based on 3D LiDAR Point Cloud

RangeNet++: Fast and Accurate LiDAR Semantic Segmentation

Real-time Progressive 3D Semantic Segmentation for Indoor Scenes

SnapNet-R: Consistent 3D Multi-View Semantic Labeling for Robotics

SqueezeSeg: Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud

SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud

SqueezeSegV3: Spatially-Adaptive Convolution for Ecient Point-Cloud Segmentation

### Voxel-based

3D Semantic Segmentation with Submanifold Sparse Convolutional Networks

3DCNN-DQN-RNN: A Deep Reinforcement Learning Framework for Semantic Parsing of Large-scale 3D Point Clouds

Fully-Convolutional Point Networks for Large-Scale Point Clouds

OctNet: Learning Deep 3D Representations at High Resolutions

ScanComplete: Large-Scale Scene Completion and Semantic Segmentation for 3D Scans

SEGCloud: Semantic Segmentation of 3D Point Clouds

VV-NET: Voxel VAE Net with Group Convolutions for Point Cloud Segmentation

### Point cloud based

#### Based on MLP

3D Recurrent Neural Networks with Context Fusion for Point Cloud Semantic Segmentation

Exploring Spatial Context for 3D Semantic Segmentation of Point Clouds

Know What Your Neighbors Do: 3D Semantic Segmentation of Point Clouds

PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space

PointSIFT: A SIFT-like Network Module for 3D Point Cloud Semantic Segmentation

PointWeb: Enhancing Local Neighborhood Features for Point Cloud Processing

#### Based on Point Conv

A-CNN: Annularly Convolutional Neural Networks on Point Clouds

Deep Parametric Continuous Convolutional Neural Networks

Dilated Point Convolutions: On the Receptive Field Size of Point Convolutions on 3D Point Clouds
Flex-Convolution

KPConv: Flexible and Deformable Convolution for Point Clouds

Monte Carlo Convolution for Learning on Non-Uniformly Sampled Point Clouds

PointCNN: Convolution On X-Transformed Points

PointConv: Deep Convolutional Networks on 3D Point Clouds

Pointwise Convolutional Neural Networks

PolarNet: An Improved Grid Representation for Online LiDAR Point Clouds Semantic Segmentation

RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

Recurrent Slice Networks for 3D Segmentation of Point Clouds

#### Based on Graph Conv

3DContextNet: K-d Tree Guided Hierarchical Learning of Point Clouds Using Local and Global Contextual Cues

DeepGCNs: Can GCNs Go as Deep as CNNs?

Dynamic Graph CNN for Learning on Point Clouds

Hierarchical Depthwise Graph Convolutional Neural Network for 3D Semantic Segmentation of Point Clouds

Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs

Global Context Reasoning for Semantic Segmentation of 3D Point Clouds

Octree guided CNN with Spherical Kernels for 3D Point Clouds

Point Attention Network for Semantic Segmentation of 3D Point Clouds

Point Clouds Learning with Attention-based Graph Convolution Networks

Spherical Kernel for Efficient Graph Convolution on 3D Point Clouds

TGNet: Geometric Graph CNN on 3-D Point Cloud Segmentation

### Multimodal based

3DMV: Joint 3D-Multi-View Prediction for 3D Semantic Scene Segmentation

A Unified Point-Based Framework for 3D Segmentation

Multi-view PointNet for 3D Scene Understanding

Point-Voxel CNN for Efficient 3D Deep Learning

Sensor Fusion for Joint 3D Object Detection and Semantic Segmentation

### Based on tangent images/lattice

Tangent Convolutions for Dense Prediction in 3D

SPLATNet: Sparse Lattice Networks for Point Cloud Processing

LatticeNet: Fast Spatio-Temporal Point Cloud Segmentation Using Permutohedral Lattices

###7.ADAS:主动自适应分割(ADAS)基线

ADAS: A Simple Active-and-Adaptive Baseline for Cross-Domain 3D Semantic Segmentation

[[Code]](https://github.com/Fayeben/ADAS)

### 8. LiDAR-based adaptive approach for semantic segmentation

Fake it, Mix it, Segment it: Bridging the Domain Gap Between Lidar Sensors

### 9.**P2Net:一种**轻量级的后处理方法来提炼点云序列的语义分割结果

P2Net: A Post-Processing Network for Refining Semantic Segmentation of LiDAR Point Cloud based on Consistency of Consecutive Frames

## 三、3D example segmentation

### 1.**Proposal Based**

3D-MPA: Multi Proposal Aggregation for 3D Semantic Instance Segmentation

3D-SIS: 3D Semantic Instance Segmentation of RGB-D Scans

GSPN: Generative Shape Proposal Network for 3D Instance Segmentation in Point Cloud

End-to-end 3D Point Cloud Instance Segmentation without Detection

Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds

SGPN: Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation

## 2.**Proposal Free**

3D Bird's-Eye-View Instance Segmentation

3D Graph Embedding Learning with a Structure-aware Loss Function for Point Cloud Semantic Instance Segmentation

3D Instance Segmentation via Multi-Task Metric Learning

Associatively Segmenting Instances and Semantics in Point Clouds

End-to-end 3D Point Cloud Instance Segmentation without Detection

JSIS3D: Joint Semantic-Instance Segmentation of 3D Point Clouds with Multi-Task Pointwise Networks and Multi-Value Conditional Random Fields

MASC: Multi-scale Affinity with Sparse Convolution for 3D Instance Segmentation

OccuSeg: Occupancy-aware 3D Instance Segmentation

PanopticFusion: Online Volumetric Semantic Mapping at the Level of Stuff and Things

PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation

## 四、3D panoramic segmentation




## 五、3D part segmentation

### 1. Rule-based data

3D Shape Segmentation with Projective Convolutional Networks

Embedding 3D Geometric Features for Rigid Object Part Segmentation

PointGrid: A Deep Network for 3D Shape Understanding

VoxSegNet: Volumetric CNNs for Semantic Part Segmentation of 3D Shapes

### 2. Based on irregular data

3D Point Capsule Networks

Escape from Cells: Deep Kd-Networks for the Recognition of 3D Point Cloud Models

FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis

MeshCNN: A Network with an Edge

Mining Point Cloud Local Structures by Kernel Correlation and Graph Pooling

O-CNN: Octree-based Convolutional Neural Networks for 3D Shape Analysis

PartNet: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation

SO-Net: Self-Organizing Network for Point Cloud Analysis

SpiderCNN: Deep Learning on Point Sets with Parameterized Convolutional Filters

SyncSpecCNN: Synchronized Spectral CNN for 3D Shape Segmentation

Directionally Convolutional Networks for 3D Shape Segmentation

