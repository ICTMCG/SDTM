<div align=center>
  
# **[CVPR 2025]** Attend to Not Attended: Structure-then-Detail Token Merging for Post-training DiT Acceleration

<p>
<a href='https://arxiv.org/pdf/2505.11707'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://github.com/ICTMCG/SDTM'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
</p>

</div>

## 🔥 News

* `2025/09/28` 🚀🚀 We update an improved version that integrates seamlessly with the diffusers StableDiffusion3Pipeline, requiring no modifications to the original diffusers code. This version removes the dependency on attention maps and is fully compatible with xFormers.

##  Dependencies
``` cmd
Python>=3.9
CUDA>=11.8
```

## 🛠 Installation

``` cmd
git clone https://github.com/ICTMCG/SDTM.git
```

### Environment Settings

#### Models and Datasets

We evaluated our model based on the Hugging Face diffusers library. You can download the related models and datasets from the following links:

Links:

| Name |                     urls                     |
| :--------------: | :------------------------------------------: |
|       COCO2017       |  http://images.cocodataset.org   |
|       PartiPrompts       |  https://github.com/google-research/parti   |
|       stabilityai/stable-diffusion-3-medium        |   https://huggingface.co/stabilityai/stable-diffusion-3-medium    |
|     stabilityai/stable-diffusion-3.5-large     | https://huggingface.co/stabilityai/stable-diffusion-3.5-large |
|     stabilityai/stable-diffusion-3.5-large-turbo     |    https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo    |

Besides, we provide a replica for our environment here:

#### Environmetns (recommended)

  ```bash
  cd SDTM
  conda env create -f environment-sdtm.yml
  ```

</details>

## 🚀 Demo and Inference

### Run DiT-ToCa

#### DDPM-250 Steps

sample images for **visualization**

```bash
bash demo.sh
```

sample images for **evaluation**

```bash
python sample.py \
  --caption-path "longest_captions.json" \
  --model-path "../../checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers" \
  --output-path "samples" \
  --height 1024 --width 1024 \
  --num_inference_steps 50 --guidance-scale 7.0 \
  --batch-size 4 --seed 0 \
  --tore-type SDTM
```

multi-GPU image sampling for **evaluation**

```bash

torchrun --nproc_per_node=4 sample_ddp.py \
  --caption-path "longest_captions.json" \
  --model-path "../../checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers" \
  --output-path "samples" \
  --height 1024 --width 1024 \
  --num_inference_steps 50 --guidance-scale 7.0 \
  --batch-size 4 --seed 0 \
  --tore-type SDTM
```

</details>

## 👍 Acknowledgements

- Thanks to [diffusers](https://github.com/huggingface/diffusers) for their excellent work and the codebase upon which we build SDTM.  
- Thanks to [ToMeSD](https://github.com/dbolya/tomesd) for their contribution of the base token merging method.  
- Thanks to [ALGM](https://github.com/tue-mps/algm-segmenter) for their work, which inspired our structure-then-detail token merging approach.  

## 📌 Citation

```bibtex
@inproceedings{fang2025attend,
  title={Attend to Not Attended: Structure-then-Detail Token Merging for Post-training DiT Acceleration},
  author={Fang, Haipeng and Tang, Sheng and Cao, Juan and Zhang, Enshuo and Tang, Fan and Lee, Tong-Yee},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={18083--18092},
  year={2025}
}
```

## :e-mail: Contact

If you have any questions, please email [`fanghaipeng21s@ict.ac.cn`](mailto:fanghaipeng21s@ict.ac.cn).
