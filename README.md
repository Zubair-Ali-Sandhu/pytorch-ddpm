# Pixel Genesis — DDPM from Scratch

> A full Denoising Diffusion Probabilistic Model implemented from scratch in pure PyTorch.
> Generates novel images by learning to reverse a fixed Gaussian noising process — pixels emerge from noise.

**Live Demo:** [DDPM Image Generator on Hugging Face Spaces](https://huggingface.co/spaces/prospect01/scratch-diffusion-implementation)

---

## Notebook

| File | Description |
|------|-------------|
| [ddpm-from-scratch.ipynb](./ddpm-from-scratch.ipynb) | Complete DDPM implementation: forward process, U-Net, training loop, sampling, and evaluation |

---

## What's Covered

| Part | Topic |
|------|-------|
| **Part 1** | Config, dataset loading, and forward diffusion (linear noise schedule) |
| **Part 2** | U-Net architecture — sinusoidal time embeddings, residual blocks, downsampling/upsampling |
| **Part 3** | Training loop with AdamW optimizer, mixed precision, and loss tracking |
| **Part 4** | Reverse sampling, image generation, reconstruction, PSNR & SSIM evaluation |

---

## Key Concepts

- **Forward process** $q(x_t | x_{t-1})$: adds Gaussian noise over $T$ timesteps using a linear $\beta$ schedule
- **Reverse process** $p_\theta(x_{t-1} | x_t)$: a U-Net learns to predict the noise $\epsilon$ at each step
- **Loss**: simplified variational lower bound — MSE between predicted and actual noise
- **Sampling**: iterative denoising from pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$ back to $x_0$

$$
\mathcal{L}_\text{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon,\; t) \|^2 \right]
$$

---

## Architecture — U-Net

```
Input (noisy image xₜ + timestep t)
    │
    ├── Encoder: Conv → [ResBlock × n, Downsample] × 3
    │           channels: 64 → 128 → 256
    │
    ├── Bottleneck: ResBlock
    │
    └── Decoder: [Upsample, ResBlock × n, skip concat] × 3
                  → Conv → output (predicted noise ε)
```

- **Time embedding:** Sinusoidal → MLP → injected into every ResBlock via affine shift
- **Skip connections:** encoder features concatenated to corresponding decoder stages

---

## Outputs

| File | Description |
|------|-------------|
| `forward_diffusion.png` | 7-step forward noising visualisation |
| `training_loss.png` | Loss vs epochs plot |
| `generated_images.png` | 5 generated samples with intermediate denoising steps |
| `reconstructed_images.png` | Input vs reconstructed comparison |

---

## Setup

```bash
git clone https://github.com/Zubair-Ali-Sandhu/pixel-genesis.git
cd pixel-genesis
pip install torch torchvision scikit-image matplotlib jupyter
jupyter notebook ddpm-from-scratch.ipynb
```

> The notebook was trained on **Kaggle** (T4 GPU). Update `Config.DATA_DIR` to point to your dataset before running locally.

---

## Author

**Zubair Ali Sandhu** — [GitHub](https://github.com/Zubair-Ali-Sandhu)
