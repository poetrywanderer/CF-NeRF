# CF-NeRF

### [Project Page](https://poetrywanderer.github.io/CF-NeRF/) | [Paper](https://arxiv.org/abs/2203.10192)

<img src="https://github.com/poetrywanderer/CF-NeRF/blob/main/image/teaser2.png?raw=true" alt="teaser" width='100%'>

This repository contains the code release for our ECCV paper: Conditional-Flow NeRF: Accurate 3D Modelling with Reliable Uncertainty Quantification. This implementation is written in Pytorch. 

## Abstract

A critical limitation of current methods based on Neural Radiance Fields (NeRF) is that they are unable to quantify the uncertainty associated with the learned appearance and geometry of the scene. This information is paramount in real applications such as medical diagnosis or autonomous driving where, to reduce potentially catastrophic failures, the confidence on the model outputs must be included into the decision-making process. In this context, we introduce Conditional-Flow NeRF (CF-NeRF), a novel probabilistic framework to incorporate uncertainty quantification into NeRF-based approaches. For this purpose, our method learns a distribution over all possible radiance fields modelling which is used to quantify the uncertainty associated with the modelled scene. In contrast to previous approaches enforcing strong constraints over the radiance field distribution, CF-NeRF learns it in a flexible and fully data-driven manner by coupling Latent Variable Modelling and Conditional Normalizing Flows. This strategy allows to obtain reliable uncertainty estimation while preserving model expressivity. Compared to previous state-of-the-art methods proposed for uncertainty quantification in NeRF, our experiments show that the proposed method achieves significantly lower prediction errors and more reliable uncertainty values for synthetic novel view and depth-map estimation.

## Installation

We recommend using Anaconda to set up the environment. Run the following commands:

```
git clone https://github.com/poetrywanderer/CF-NeRF.git
cd CF-NeRF
pip install -r requirements.txt
```

<details>
  <summary> Dependencies (click to expand) </summary>
  
  ## Dependencies
  - PyTorch 1.4
  - matplotlib
  - numpy
  - imageio
  - imageio-ffmpeg
  - configargparse
  
The LLFF data loader requires ImageMagick.

You will also need the [LLFF code](http://github.com/fyusion/llff) (and COLMAP) set up to compute poses if you want to run on your own real data.
  
</details>

## Data

Download the LLFF datasets from the NeRF official Google Drive, the LF datasets. 

### Your Own data
To play with other scenes, place your own data according to the following directory structure:
```
├── configs                                                                                                       
│   ├── ...                                                                                     
│                                                                                               
├── data                                                                                                                                                                                                       
│   ├── nerf_llff_data                                                                                                  
│   │   └── fern                                                                                                                             
│   │   └── flower  # downloaded llff dataset                                                                                   
|   |   └── ...
│   ├── lf_data  
│   │   └── basket   # downloaded llff dataset
|   |   └── ...
```

## Training

To train CF-NeRF on different datasets:

```
bash train_NF.sh # change the content there
```

To test NeRF trained on different datasets: 

```
python test_NF.sh # change the content there
```

### Pre-trained Models (Coming soon)

## License

CF-NeRF is MIT-licensed. The license applies to the pre-trained models as well.

## Citation
Please cite as:
```
@misc{CF-NeRF,
    title = {Conditional-Flow NeRF: Accurate 3D Modelling with Reliable Uncertainty Quantification},
    author = {Shen, Jianxiong and Agudo, Antonio and Moreno-Noguer, Francesc and Ruiz, Adria},
    year = {2022},
    eprint={2203.10192},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
