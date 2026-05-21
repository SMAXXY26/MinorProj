# Dataset & Method Citations

All datasets, codebases, and techniques used in the WeaponDetection V2
pipeline are listed here.  BibTeX entries are included for every source that
has a citable paper; Roboflow Universe datasets are cited in the format
recommended by Roboflow.

---

## 1 — Academic Datasets

### 1.1 The Monash Guns Dataset
CCTV handgun-detection dataset from 250 surveillance videos (5,500 frames).
Used as a primary source for pistol and rifle annotations.

```
@inproceedings{monash_guns_2019,
  title     = {Gun Detection in Surveillance Videos using Deep Neural Networks},
  author    = {Lim Jun Yi, Marcus and others},
  booktitle = {Asia-Pacific Signal and Information Processing Association Annual
               Summit and Conference (APSIPA ASC)},
  year      = {2019},
  url       = {https://github.com/MarcusLimJunYi/Monash-Guns-Dataset}
}
```
Roboflow mirror: https://universe.roboflow.com/arms/the-monash-guns-dataset

---

### 1.2 OD-WeaponDetection (Sohas / ari-dasci)
5,859 images, 6,446 annotated objects across knife and pistol classes.
Collected from YouTube surveillance footage; already in YOLO-compatible format.
Used via `merge_external_data.py` (image prefix: `sohas_*`, `kdet_*`).

```
@misc{od_weapon_detection,
  title        = {{OD-WeaponDetection}: Datasets for Weapon Detection},
  author       = {Laraba, Sohaib and others},
  year         = {2020},
  howpublished = {\url{https://github.com/ari-dasci/OD-WeaponDetection}},
  note         = {Andalusian Research Institute in Data Science and
                  Computational Intelligence (DaSCI)}
}
```

---

### 1.3 VisDrone-DET (aerial backgrounds / hard negatives)
Drone-captured aerial imagery with 10 object categories. Used as hard-negative
backgrounds to suppress drone-viewpoint false positives (image prefix:
`visdrone_*`).

```
@inproceedings{visdrone2018,
  title     = {{VisDrone-DET2018}: The Vision Meets Drone Object Detection in
               Image Challenge Results},
  author    = {Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao
               and Ling, Haibin},
  booktitle = {European Conference on Computer Vision Workshops (ECCVW)},
  year      = {2018},
  note      = {arXiv:1804.07437},
  url       = {https://github.com/VisDrone/VisDrone-Dataset}
}

@inproceedings{visdrone2019,
  title     = {{VisDrone-DET2019}: The Vision Meets Drone Object Detection in
               Image Challenge Results},
  author    = {Du, Dawei and others},
  booktitle = {IEEE/CVF International Conference on Computer Vision Workshops
               (ICCVW)},
  year      = {2019},
  url       = {https://openaccess.thecvf.com/content_ICCVW_2019/papers/VISDrone/
               Du_VisDrone-DET2019_The_Vision_Meets_Drone_Object_Detection_in_
               Image_Challenge_ICCVW_2019_paper.pdf}
}
```

---

### 1.4 Microsoft COCO (hard negatives — FP correction)
Used as the source of visually challenging weapon-free images for the FP
correction round 1 (Stage 4 of the training pipeline).

```
@inproceedings{coco2014,
  title     = {Microsoft {COCO}: Common Objects in Context},
  author    = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and
               Hays, James and Perona, Pietro and Ramanan, Deva and
               Doll{\'a}r, Piotr and Zitnick, C. Lawrence},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2014},
  note      = {arXiv:1405.0312},
  doi       = {10.1007/978-3-319-10602-1_48}
}
```

---

## 2 — Roboflow Universe Datasets

All Roboflow datasets were downloaded via the Roboflow Python SDK.  Cite the
platform as:

```
@misc{roboflow2022,
  title        = {Roboflow Universe},
  author       = {Nelson, Brad and Solawetz, Joseph and others},
  year         = {2022},
  howpublished = {\url{https://universe.roboflow.com}},
  publisher    = {Roboflow},
  note         = {visited on 2026-04-23}
}
```

Individual datasets used (workspace / project / version):

| Image prefix | Workspace | Project | Version | Class(es) |
|---|---|---|---|---|
| `rf_knife_bottle_cu_*` | laserworkspace | knife_bottle_cup | 1 | knife |
| `rf_knife-aqe*` | cao-jkghk | knife-aqe0g | 3 | knife |
| `rf_knife-sd*` | motos-and-cars | knife-sd3xq | 3 | knife |
| `rf_knife-skqnq*` | labelling-7k1aj | knife-skqnq | 1 | knife |
| `rf_rifle-xr*` | doken-edgar | rifle-xr2aa | 1 | rifle |
| `rf_handgun*` | model-training-iwx9e | handgun-longgun_surv_v2-o0rky | 2 | pistol/rifle |
| `rf_rifle-*` (frwne) | weapon-detection-frwne | rifle-1b8vx | 2 | rifle |
| `drone_weapon*` / `weapon_detection*` | weapon-detection-cctv | weapon-detection-cctv-v3-dataset | 1 | knife/pistol/rifle |
| `drone_weapon*` (m7qso) | yolov7test-u13vc | weapon-detection-m7qso | 16 | knife/pistol/rifle |
| `drone_gun*` | em2023 | gun-detection-s5poj | 1 | pistol/rifle |

Each dataset's individual BibTeX can be retrieved from its Roboflow Universe
page under "Cite this project".

---

## 3 — Model Architecture Papers

### 3.1 YOLOv8 (detector backbone — teacher)
```
@misc{yolov8_2023,
  title        = {{Ultralytics YOLOv8}},
  author       = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  year         = {2023},
  howpublished = {\url{https://github.com/ultralytics/ultralytics}},
  license      = {AGPL-3.0}
}
```

### 3.2 YOLO11 (student model)
```
@misc{yolo11_2024,
  title        = {{Ultralytics YOLO11}},
  author       = {Jocher, Glenn and Qiu, Jing},
  year         = {2024},
  howpublished = {\url{https://github.com/ultralytics/ultralytics}},
  license      = {AGPL-3.0}
}
```

### 3.3 EfficientNet (classifier — Gate 2)
```
@inproceedings{efficientnet2019,
  title     = {{EfficientNet}: Rethinking Model Scaling for Convolutional
               Neural Networks},
  author    = {Tan, Mingxing and Le, Quoc V.},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2019},
  note      = {arXiv:1905.11946}
}
```

---

## 4 — Training Method Papers

### 4.1 Knowledge Distillation
```
@inproceedings{hinton_kd_2015,
  title     = {Distilling the Knowledge in a Neural Network},
  author    = {Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  booktitle = {NIPS 2014 Deep Learning Workshop},
  year      = {2015},
  note      = {arXiv:1503.02531}
}
```

### 4.2 Attention Transfer (feature KD)
```
@inproceedings{attention_transfer_2017,
  title     = {Paying More Attention to Attention: Improving the Performance
               of Convolutional Neural Networks via Attention Transfer},
  author    = {Zagoruyko, Sergey and Komodakis, Nikos},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2017},
  note      = {arXiv:1612.03928}
}
```

### 4.3 Focal Loss (classifier training)
```
@inproceedings{focal_loss_2017,
  title     = {Focal Loss for Dense Object Detection},
  author    = {Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and
               He, Kaiming and Doll{\'a}r, Piotr},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year      = {2017},
  note      = {arXiv:1708.02002}
}
```

### 4.4 ByteTrack (multi-object tracking)
```
@inproceedings{bytetrack2022,
  title     = {{ByteTrack}: Multi-Object Tracking by Associating Every
               Detection Box},
  author    = {Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong
               and Weng, Fangyi and Yuan, Zehuan and Luo, Ping and Liu,
               Wenyu and Wang, Xinggang},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022},
  note      = {arXiv:2110.06864}
}
```

---

## 5 — Small-Object Augmentation & Detection Papers

These papers motivate the techniques implemented in
`src/functions/small_object_aug.py` and `config/yolo11n_p2.yaml`.

### 5.1 YOLOv4 — Mosaic augmentation
Introduced mosaic (4-image composite) and copy-paste as "bag-of-freebies"
that significantly improve small-object recall.

```
@misc{yolov4_2020,
  title  = {{YOLOv4}: Optimal Speed and Accuracy of Object Detection},
  author = {Bochkovskiy, Alexey and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
  year   = {2020},
  note   = {arXiv:2004.10934}
}
```

### 5.2 Simple Copy-Paste
Motivates the `small_object_copy_paste` function: pasting small weapon crops
onto background images to synthesise training diversity.

```
@inproceedings{copy_paste_2021,
  title     = {Simple Copy-Paste Is a Strong Data Augmentation Method for
               Instance Segmentation},
  author    = {Ghiasi, Golnaz and Cui, Yin and Srinivas, Aravind and
               Qian, Rui and Lin, Tsung-Yi and Cubuk, Ekin D. and
               Le, Quoc V. and Zoph, Barret},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern
               Recognition (CVPR)},
  year      = {2021},
  note      = {arXiv:2012.07177}
}
```

### 5.3 SAHI — Slicing Aided Hyper Inference
Tiling-based inference strategy that substantially improves detection AP on
small objects in aerial imagery.  Used as inspiration for the tiled training
patch approach.

```
@inproceedings{sahi_2022,
  title     = {Slicing Aided Hyper Inference and Fine-tuning for Small
               Object Detection},
  author    = {Akyon, Fatih Cagatay and Altinuc, Sinan Onur and
               Temizel, Alptekin},
  booktitle = {IEEE International Conference on Image Processing (ICIP)},
  year      = {2022},
  note      = {arXiv:2202.06934},
  url       = {https://github.com/obss/sahi}
}
```

### 5.4 Scale Match for Tiny Person Detection
Motivates scale-aware sampling: aligning the distribution of object scales
between pre-training data and the target domain improves tiny-object AP.

```
@inproceedings{scale_match_2019,
  title     = {Scale Match for Tiny Person Detection},
  author    = {Yu, Jingtao and others},
  booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2020},
  note      = {arXiv:1912.10664}
}
```

### 5.5 DOTA — Aerial Object Detection Dataset
Benchmark for aerial object detection that established best-practice
augmentation strategies (multi-scale training, rotation, flipping) for
drone-mounted sensors.

```
@inproceedings{dota_2018,
  title     = {{DOTA}: A Large-scale Dataset for Object Detection in Aerial
               Images},
  author    = {Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhuotao
               and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and
               Pelillo, Marcello and Zhang, Liangpei},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition
               (CVPR)},
  year      = {2018},
  note      = {arXiv:1711.10398},
  url       = {https://captain-whu.github.io/DOTA/}
}
```

### 5.6 P2 Shallow Feature Fusion (implementation basis)
The `config/yolo11n_p2.yaml` architecture adds a stride-4 detection head that
fuses the backbone P2 feature map (160×160 at 640 px input) before spatial
detail is lost.  This directly follows the multi-scale head design discussed in:

```
@inproceedings{fpn_2017,
  title     = {Feature Pyramid Networks for Object Detection},
  author    = {Lin, Tsung-Yi and Doll{\'a}r, Piotr and Girshick, Ross and
               He, Kaiming and Hariharan, Bharath and Belongie, Serge},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition
               (CVPR)},
  year      = {2017},
  note      = {arXiv:1612.03144}
}
```

---

## 6 — Dataset Statistics (as of 2026-04-23)

| Split | Images | Annotations | Knife | Pistol | Rifle |
|---|---|---|---|---|---|
| train | 49,224 | ~58,903 | ~17,152 | ~25,504 | ~16,247 |
| val | 13,042 | ~13,042 | ~3,809 | ~5,738 | ~3,495 |
| **Total** | **62,266** | **71,945** | **20,961** | **31,242** | **19,742** |

Background (empty label) files: 2,011 (hard-negative images for FP correction).
Very small objects (√(w·h) < 0.04): 291 annotations — primary target of the
loss-guided augmentation.

Label quality: 0 out-of-bounds coordinates, 0 known duplicates after the
cleanup procedure documented in CLAUDE.md.
