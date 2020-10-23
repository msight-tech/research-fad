[![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc/4.0/)

# Representation Sharing for Fast Object Detector Search and Beyond 

This is the **PyTorch** implementation for the paper:

    Representation Sharing for Fast Object Detector Search and Beyond;
    Y. Zhong, Z. Deng, S. Guo, M. R. Scott, W. Huang 
    In: Proc. European Conference on Computer Vision (ECCV), 2020.
    arXiv preprint arXiv:2007.12075 

The full paper is available at: [https://arxiv.org/abs/2007.12075](https://arxiv.org/abs/2007.12075). 

## Highlights
- **Support architecture search:** This implementation supports both the architecture search and the training of the searched detectors.
- **Efficient search:** FAD is an order of magnitude faster than previous object detector search methods. A complete search on Pascal VOC 2012 with [FCOS](https://arxiv.org/abs/1904.01355) only takes 0.6 GPU-days.
- **Diverse operations:** FAD contains up to 12 candidate operations in the search space, providing more combinations to be explored. 
- **Better performance:** The searched FAD components can improve the detection performance by up to 1.7 AP over their counterparts on MS-COCO. 


## Hardware
- For the architecture search, 1 Nvidia Titan Xp GPU is used. Multi-GPU search is also supported.
- For the training (with searched architectures), 8 Nvidia Titan Xp GPUs are used. 


## Datasets
Two datasets are required.
- [MS-COCO 2017](https://cocodataset.org/#download) 
- [VOC 2012](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) (for the architecture search)

The two datasets are expected to be placed in the following structure:
```
FAD/
  datasets/
    coco/
      images/
      annotations/
        instances_{train,val}2017.json  
    voc/
      VOC2012/
        Annotations/
        ImageSets/
        JPEGImages/
```

## Installation
This FAD implementation is based on [FCOS](https://github.com/tianzhi0549/FCOS) and [DARTS](https://github.com/khanrc/pt.darts). 
The installation of FAD is mostly the same as [FCOS](https://github.com/tianzhi0549/FCOS), with two additional packages in the requirements: 
- tensorboardX 
- graphviz 

The full installation instructions of FAD refer to [INSTALL.md](INSTALL.md).


## Architecture search 

The following command line will start the search for the subsnetworks with FCOS on Pascal VOC 2012:

    CUDA_VISIBLE_DEVICES=0 \
    python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=$((RANDOM + 10000)) \
        tools/search_net.py \
        --skip-test \
        --config-file configs/fad/search/fad-fcos_imprv_R_50_FPN_1x.yaml \
        --use-tensorboard \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/search/fad-fcos_imprv_R_50_FPN_1x

Notes:
1) If out-of-GPU-memory issue occurs, you can conduct the search using more than one GPU. For example, you can simply change `CUDA_VISIBLE_DEVICES=0` to `CUDA_VISIBLE_DEVICES=0,1,2,3` to start the multi-GPU search, where `0,1,2,3` are the IDs of the GPUs to be used.
2) The genotypes of the derived architectures will be saved into `OUTPUT_DIR`, as a file named *genotype.log*. The first genotype is for bounding box regreesion subnetwork, and the second is for the classification subnetwork. This *genotype.log* will be loaded during training and evaluation.


## Training with searched architectures 

The following command line will train FCOS_imprv_R_50_FPN_1x with the searched subnetworks on 8 GPUs:

    python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --config-file configs/fad/augment/fad-fcos_imprv_R_50_FPN_1x.yaml \
        --genotype-file training_dir/search/fad-fcos_imprv_R_50_FPN_1x/genotype.log \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/augment/fad-fcos_imprv_R_50_FPN_1x
        
Notes:
1) The models will be saved into `OUTPUT_DIR`.
2) If you want to augment using different backbone networks, please change the file specified by `--config-file`.
3) If you want to search or train detectors on your own dataset, please prepare your dataset (including images and annotation files) in the same structure as MS-COCO, and change the config file accordingly.


## Evaluation

The command line below evaluate the trained model on MS-COCO minival split:

    python -m torch.distributed.launch \
        tools/test_net.py \
        --config-file configs/fad/augment/fad-fcos_imprv_R_50_FPN_1x.yaml \
        --genotype-file training_dir/search/fad-fcos_imprv_R_50_FPN_1x/genotype.log \
        MODEL.WEIGHT path_to_the_weights.pth \
        TEST.IMS_PER_BATCH 4

Notes:
1) Please replace `path_to_the_weights.pth` with your own trained model weights. For example, *training_dir/augment/fad-fcos_imprv_R_50_FPN_1x/model_final.pth*.
2) The weights file must match the detector specified by the config file and genotype file.
3) The inference can be done using multiple GPUs. You can modify the command line in a similar way as augment.


## Results

The following results are based on the searched architectures displayed in the paper.

Model | Channel size | Multi-scale training | AP (minival) | AP (test-dev) 
--- |:---:|:---:|:---:|:---:
fad-fcos_imprv_R_50_FPN_1x | 96 | No | 40.3 | -  
fad-fcos_imprv_R_101_FPN_2x | 96 | Yes | 44.2 | 44.1
fad-fcos_imprv_R_101_64x4d_FPN_2x | 128 | Yes | 46.0 | 46.0 




## Explored architectures

For object detection:

    # 1st row: bounding box regression subnetwork
    # 2nd row: classification subnetwork

    # --- Architectures reported in the paper
    Genotype(normal=[[('sinSep_conv_3x3_x3', 0), ('std_conv_3x3_x3', 1)], [('std_conv_3x3', 2), ('sinSep_conv_3x3+dil_conv_3x3', 1)], [('sinSep_conv_3x3_x2', 2), ('std_conv_3x3_x2', 3)]], normal_concat=range(2, 5))
    Genotype(normal=[[('sinSep_conv_3x3_x2+dil_conv_3x3', 1), ('std_conv_3x3_x3', 0)], [('sinSep_conv_3x3_x2+dil_conv_3x3', 0), ('sinSep_conv_3x3', 2)], [('sinSep_conv_3x3_x2', 2), ('std_conv_3x3', 3)]], normal_concat=range(2, 5))

    # --- Other good architectures 
    Genotype(normal=[[('std_conv_3x3_x3', 0), ('std_conv_3x3_x2', 1)], [('std_conv_3x3', 0), ('std_conv_3x3+dil_conv_3x3', 2)], [('std_conv_3x3_x2+dil_conv_3x3', 3), ('std_conv_3x3+dil_conv_3x3', 2)]], normal_concat=range(2, 5))
    Genotype(normal=[[('std_conv_3x3_x3', 1), ('sinSep_conv_3x3+dil_conv_3x3', 0)], [('sinSep_conv_3x3', 0), ('sinSep_conv_3x3+dil_conv_3x3', 2)], [('std_conv_3x3', 1), ('std_conv_3x3_x3', 0)]], normal_concat=range(2, 5))

## Citations
Please cite our paper if this implementation helps your research. BibTeX reference is shown in the following.
```
@inproceedings{zhong2020representation,
  title={Representation Sharing for Fast Object Detector Search and Beyond}, 
  author={Zhong, Yujie and Deng, Zelu and Guo, Sheng and Scott, Matthew R and Huang, Weilin}, 
  booktitle={In: Proceedings of the European Conference on Computer Vision (ECCV)},  
  year={2020} 
  }
```

## Contact

For any questions, please feel free to reach: 
```
github@malongtech.com
```

## License

FAD is CC-BY-NC 4.0 licensed, as found in the [LICENSE](LICENSE) file. It is released for academic research / non-commercial use only. If you wish to use for commercial purposes, please contact sales@malongtech.com.

