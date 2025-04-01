# 🦴 AnyBoneSegmenter 


**AnyBoneSegmenter** is a deep learning-based medical image segmentation model specialized in accurately segmenting diverse bone structures across **MRI** and **CT** modalities. 

**SegmentAnyBone** is a foundational model-based bone segmentation algorithm adapted from **Segment Anything Model (SAM)** for MRI scans.

## 🌟 Features
- ✅ **Multi-modality support**: Works with both **MRI** and **CT** scans
- ✅ **17 Body Parts Segmentation**: Covers major anatomical regions
- ✅ **SAM-based Architecture**: Leverages the power of Segment Anything Model
- ✅ **Medical-Grade Accuracy**: Optimized for clinical use cases

## 📋 Supported Body Parts
| Region 1       | Region 2         | Region 3        |
|----------------|------------------|-----------------|
| Humerus       | Thoracic Spine   | Lumbar Spine    |
| Forearm       | Pelvis           | Hand            |
| Lower Leg     | Shoulder         | Chest           |
| Arm           | Elbow            | Hip             |
| Wrist         | Thigh            | Knee            |
| Foot          | Ankle            |                 |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (if using GPU)


## 📝 Citation
If you use AnyBoneSegmenter in your research, please cite:

```bibtex
@misc{gu2024segmentanybone,
  title={SegmentAnyBone: A Universal Model that Segments Any Bone at Any Location on MRI}, 
  author={Hanxue Gu and Roy Colglazier and Haoyu Dong and Jikai Zhang and Yaqian Chen and Zafer Yildiz and Yuwen Chen and Lin Li and Jichen Yang and Jay Willhite and Alex M. Meyer and Brian Guo and Yashvi Atul Shah and Emily Luo and Shipra Rajput and Sally Kuehn and Clark Bulleit and Kevin A. Wu and Jisoo Lee and Brandon Ramirez and Darui Lu and Jay M. Levin and Maciej A. Mazurowski},
  year={2024},
  eprint={2401.12974},
  archivePrefix={arXiv},
  primaryClass={eess.IV}
}
