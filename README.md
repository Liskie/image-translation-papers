# 1. Medical Image Translation

# 1.1 Paired Image Translation

## Image-to-Image Translation with Conditional Adversarial Networks (2017)

**Authors**: Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros  
**Venue**: IEEE Conference on Computer Vision and Pattern Recognition (CVPR)  
**Link**: [https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf)

### 🧠 Method

- **Key Technique**: Conditional Generative Adversarial Network (cGAN)
- **Type**: Paired
- **Architecture Notes**: Utilizes a U-Net-based generator and a PatchGAN discriminator to learn a mapping from input to output images conditioned on paired data.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various (e.g., grayscale images)
- **Target Modality**: Corresponding translated images (e.g., colorized images)
- **Paired/Unpaired Data**: Paired

### 🌟 Highlights

- Demonstrates the versatility of cGANs in performing various image-to-image translation tasks.
- Emphasizes the importance of paired datasets in achieving high-quality translations.

### 📊 Evaluation

- **Metrics Used**: Mean Squared Error (MSE), Structural Similarity Index (SSIM)
- **Datasets**: Facades, Cityscapes, and others
- **Baseline Comparison**: Compared against traditional methods for each specific task

### 📌 Summary Notes

- Introduces the pix2pix framework, which has become foundational in paired image translation tasks.
- Provides a general-purpose solution applicable to various domains, including medical imaging.

## Paired-Unpaired Unsupervised Attention Guided GAN with Transfer Learning for Medical Image Translation (2021)

**Authors**: Alaa Abu-Srhan, Israa Almallahi, Mohammad A.M. Abushariah, Waleed Mahafza, Omar S. Al-Kadi
**Venue**: Computers in Biology and Medicine  
**Link**: [https://www.sciencedirect.com/science/article/abs/pii/S0010482521005576](https://www.sciencedirect.com/science/article/abs/pii/S0010482521005576)

### 🧠 Method

- **Key Technique**: Attention Guided Generative Adversarial Network (AGGAN) with Transfer Learning
- **Type**: Paired and Unpaired
- **Architecture Notes**: Utilizes attention mechanisms to focus on relevant regions during translation and incorporates transfer learning to enhance performance.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various medical imaging modalities
- **Target Modality**: Corresponding translated modalities
- **Paired/Unpaired Data**: Both paired and unpaired

### 🌟 Highlights

- Alleviates the need for strictly paired datasets by effectively utilizing both paired and unpaired data.
- Demonstrates improved performance in medical image translation tasks through attention mechanisms.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: -
- **Baseline Comparison**: Compared against existing GAN-based methods

### 📌 Summary Notes

- Proposes a flexible framework capable of handling both paired and unpaired data.
- Emphasizes the importance of attention mechanisms in focusing on critical regions during translation.

## Image Translation for Medical Image Generation: Ischemic Stroke Lesion Segmentation (2022)

**Authors**: Moritz Platscher, Jonathan Zopes, Christian Federau
**Venue**: Biomedical Signal Processing and Control  
**Link**: [https://www.sciencedirect.com/science/article/pii/S1746809421008806](https://www.sciencedirect.com/science/article/pii/S1746809421008806)

### 🧠 Method

- **Key Technique**: Pix2Pix
- **Type**: Paired
- **Architecture Notes**: Implements the Pix2Pix framework for translating between different medical imaging modalities to aid in ischemic stroke lesion segmentation.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: CT images
- **Target Modality**: MRI-like images
- **Paired/Unpaired Data**: Paired

### 🌟 Highlights

- Demonstrates the effectiveness of Pix2Pix in generating synthetic MRI images from CT scans.
- Aids in improving ischemic stroke lesion segmentation by providing additional imaging information.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Ischemic stroke patient datasets
- **Baseline Comparison**: Compared against traditional segmentation methods

### 📌 Summary Notes

- Highlights the potential of image translation techniques in enhancing medical image analysis tasks.
- Suggests that synthetic images can provide complementary information for better diagnosis and treatment planning.

## Overview of image-to-image translation by use of deep neural networks: denoising, super-resolution, modality conversion, and reconstruction in medical imaging (2019)

**Authors**: Shizuo Kaji, Satoshi Kida  
**Venue**: Radiological Physics and Technology  
**Link**: [https://arxiv.org/abs/1905.08603](https://arxiv.org/abs/1905.08603)

### 🧠 Method

- **Key Technique**: Combination of AUTOMAP and Pix2Pix
- **Type**: Paired
- **Architecture Notes**: Combines AUTOMAP (which learns domain-agnostic mappings via a fully connected layer followed by CNNs) with Pix2Pix to apply conditional GANs for tasks like super-resolution, denoising, and modality translation.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Noisy or low-res CT/MRI
- **Target Modality**: Clean, high-res, or different modality image
- **Paired/Unpaired Data**: Paired

### 🌟 Highlights

- Demonstrates effectiveness of combining domain-agnostic mapping (AUTOMAP) with the strong generative performance of GANs (Pix2Pix).
- Explores multiple applications: denoising, super-resolution, CT-to-MRI translation.

### 📊 Evaluation

- **Metrics Used**: PSNR, SSIM
- **Datasets**: Publicly available clinical CT/MRI datasets
- **Baseline Comparison**: Compared to classical interpolation and CNN-only models

### 📌 Summary Notes

- One of the early efforts to combine learning-based reconstruction and translation.
- Image-to-Image Translation by CNNs Trained on Paired Data (AUTOMAP + Pix2Pix) on GitHub
- Emphasizes multi-purpose paired translation using a hybrid pipeline.

## Swin Transformer-Based GAN for Multi-Modal Medical Image Translation (2022)

**Authors**: Shouang Yan, Chengyan Wang, Weibo Chen, Jun Lyu  
**Venue**: Frontiers in Oncology
**Link**: [https://www.frontiersin.org/articles/10.3389/fnins.2022.939518/full](https://www.frontiersin.org/articles/10.3389/fnins.2022.939518/full)

### 🧠 Method

- **Key Technique**: Swin Transformer-based Generative Adversarial Network (GAN)
- **Type**: Paired
- **Architecture Notes**: Incorporates Swin Transformer modules within the generator and registration network to capture global features and model long-distance interactions.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various MRI sequences (e.g., T1-weighted)
- **Target Modality**: Corresponding translated MRI sequences (e.g., T2-weighted)
- **Paired/Unpaired Data**: Paired

### 🌟 Highlights

- Addresses the limitations of traditional CNN-based models in capturing global contextual information.
- Demonstrates superior performance in translating between different MRI modalities.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Public and clinical MRI datasets
- **Baseline Comparison**: Compared against traditional GAN-based methods

### 📌 Summary Notes

- Highlights the potential of transformer-based architectures in medical image translation tasks.
- Suggests that capturing long-range dependencies is crucial for accurate modality translation.

# 1.2 Unpaired Image Translation

## Bridging the Gap Between Paired and Unpaired Medical Image Translation (2021)

**Authors**: Pauliina Paavilainen, Saad Ullah Akram, Juho Kannala  
**Venue**: arXiv  
**Link**: [https://arxiv.org/abs/2110.08407](https://arxiv.org/abs/2110.08407)

### 🧠 Method

- **Key Technique**: Modified pix2pix for unpaired data
- **Type**: Unpaired
- **Architecture Notes**: Introduces modifications to the pix2pix model to handle unpaired CT and MR data, utilizing MRCAT pairs generated from MR scans to ensure alignment between input and translated images.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: CT or MR images
- **Target Modality**: MR or CT images
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Addresses the challenge of translating between CT and MR images without paired datasets.
- Utilizes synthetic MRCAT pairs to bridge the gap between unpaired datasets.
- Demonstrates improved performance over baseline pix2pix, pix2pixHD, and CycleGAN models in terms of FID and KID scores.

### 📊 Evaluation

- **Metrics Used**: Fréchet Inception Distance (FID), Kernel Inception Distance (KID)
- **Datasets**: Unpaired CT and MR datasets with MRCAT pairs
- **Baseline Comparison**: Compared against pix2pix, pix2pixHD, and CycleGAN

### 📌 Summary Notes

- Proposes a novel approach to unpaired medical image translation by leveraging synthetic paired data.
- Highlights the potential of integrating synthetic data to enhance unpaired translation models.

## ContourDiff: Unpaired Image-to-Image Translation with Structural Consistency for Medical Imaging (2024)

**Authors**: Yuwen Chen, Nicholas Konz, Hanxue Gu, et al.  
**Venue**: arXiv  
**Link**: [https://arxiv.org/abs/2403.10786](https://arxiv.org/abs/2403.10786)

### 🧠 Method

- **Key Technique**: Diffusion models with anatomical contour constraints
- **Type**: Unpaired
- **Architecture Notes**: Leverages domain-invariant anatomical contour representations to guide a diffusion model, ensuring preservation of anatomical structures during translation.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: CT images
- **Target Modality**: MRI images
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Introduces a novel metric to quantify structural bias between domains.
- Ensures anatomical structure preservation during unpaired translation.
- Outperforms existing methods in lumbar spine and hip-and-thigh CT-to-MRI translation tasks.

### 📊 Evaluation

- **Metrics Used**: Segmentation performance on translated images, Fréchet Inception Distance (FID), Kernel Inception Distance (KID)
- **Datasets**: Lumbar spine and hip-and-thigh CT and MRI datasets
- **Baseline Comparison**: Compared against other unpaired image translation methods

### 📌 Summary Notes

- Emphasizes the importance of structural consistency in medical image translation.
- Demonstrates the effectiveness of integrating anatomical constraints into diffusion models for unpaired translation tasks.

## Target-Guided Diffusion Models for Unpaired Cross-Modality Medical Image Translation (2024)

**Authors**: Yimin Luo, Qinyu Yang, Ziyi Liu, et al.  
**Venue**: IEEE Journal of Biomedical and Health Informatics  
**Link**: [https://pubmed.ncbi.nlm.nih.gov/38662561/](https://pubmed.ncbi.nlm.nih.gov/38662561/)

### 🧠 Method

- **Key Technique**: Target-guided diffusion model (TGDM)
- **Type**: Unpaired
- **Architecture Notes**: Employs a perception-prioritized weight scheme during training and utilizes a pre-trained classifier in the reverse process to mitigate modality-specific remnants from source data.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: MRI or CT images
- **Target Modality**: CT or MRI images
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Addresses the challenge of synthesizing target medical images without paired data.
- Demonstrates the effectiveness of diffusion models in generating realistic anatomical structures.
- Validated through subjective assessments indicating clinical relevance.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Brain MRI-CT and prostate MRI-US datasets
- **Baseline Comparison**: -

### 📌 Summary Notes

- Highlights the potential of diffusion models in unpaired medical image translation.
- Emphasizes the importance of perception-prioritized training and guided sampling in achieving realistic translations.

## Unpaired Medical Image Translation Based on WaveUNIT (2024)

**Authors**: Lingfeng Li, Qingling Cai
**Venue**: International Conference on Algorithms, High Performance Computing, and Artificial Intelligence (AHPCAI 2024)
**Link**: [https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13403/1340317/Unpaired-medical-image-translation-based-on-WaveUNIT/10.1117/12.3051578.short](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13403/1340317/Unpaired-medical-image-translation-based-on-WaveUNIT/10.1117/12.3051578.short)

### 🧠 Method

- **Key Technique**: Wavelet-based Unsupervised Image-to-Image Translation (WaveUNIT)
- **Type**: Unpaired
- **Architecture Notes**: Integrates wavelet transform with UNIT framework to capture multi-scale features and enhance translation quality.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various medical imaging modalities
- **Target Modality**: Corresponding translated modalities
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Addresses limitations of existing GAN-based models in medical image translation.
- Demonstrates improved performance by capturing both global and local features through wavelet decomposition.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: -
- **Baseline Comparison**: Compared against traditional GAN-based methods

### 📌 Summary Notes

- Proposes a novel integration of wavelet transform with UNIT for unpaired medical image translation.
- Suggests that multi-scale feature capture is crucial for accurate and realistic translations.

## Unpaired Image-to-Image Translation with Density Changing Regularization (2022)

**Authors**: Shaoan Xie, Qirong Ho, Kun Zhang  
**Venue**: Advances in Neural Information Processing Systems (NeurIPS)  
**Link**: [https://proceedings.neurips.cc/paper_files/paper/2022/file/b7032a9d960ebb6bcf1ce9d73b5861f0-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/b7032a9d960ebb6bcf1ce9d73b5861f0-Paper-Conference.pdf)

### 🧠 Method

- **Key Technique**: Density Changing Regularization (DCR)
- **Type**: Unpaired
- **Architecture Notes**: Builds on CycleGAN with a new regularization that explicitly reduces density shifts between source and target distributions.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Unpaired natural or medical images (CT, MRI)
- **Target Modality**: Translated modality (e.g., CT → MRI)
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- The DCR module significantly stabilizes training and improves realism of the translated images.
- Particularly useful when source and target domains differ significantly in texture or anatomy.

### 📊 Evaluation

- **Metrics Used**: FID, KID, pixel-wise accuracy, segmentation performance
- **Datasets**: BraTS, IXI, ADNI
- **Baseline Comparison**: Outperforms CycleGAN, CUT, MUNIT in structure preservation and detail clarity.

### 📌 Summary Notes

- Provides a theoretically grounded enhancement to CycleGAN-style training.
- Easily integrable with other unpaired models to improve domain adaptation.

## Structure-Preserving Diffusion Model for Unpaired Medical Image Translation (2025)

**Authors**: Haoshen Wang, Xiaodong Wang, Zhiming Cui  
**Venue**: MLMI 2024 (Lecture Notes in Computer Science, LNCS 15241)  
**Link**: [https://doi.org/10.1007/978-3-031-73284-3_22](https://doi.org/10.1007/978-3-031-73284-3_22) | [GitHub](https://github.com/Mors20/Structure-Preserving-Unpaired-Medical-Image-Translation)

### 🧠 Method

- **Key Technique**: DDPM with Interleaved Sampling Refinement (ISR)
- **Type**: Unpaired
- **Architecture Notes**: Conditional UNet diffusion model with edge (Canny) guidance and attention-based edge reshaping; ISR alternates edge-conditioned and unconditioned steps.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: CT
- **Target Modality**: MR (T2-weighted)
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Preserves anatomical structure using edge (Canny) conditioning
- No need for paired data
- Introduces **Interleaved Sampling Refinement (ISR)** to balance structural fidelity and modality realism
- Outperforms CycleGAN, MUNIT, and prior diffusion methods on both structure and quality
- Uses attention maps to guide realistic edge regeneration

### 📊 Evaluation

- **Metrics Used**:
    - Dice coefficient, HD95 (AMOS – via segmentation performance)
    - PSNR, SSIM (Pelvic dataset – paired CT-MR)
- **Datasets**:
    - AMOS (Abdomen CT/MR)
    - Gold Atlas Pelvic (CT, T1, T2 MR)
- **Baseline Comparison**:  
  CycleGAN, MUNIT, CycleGAN-Lcc, FGDM

### 📌 Summary Notes

- **Strengths**: Excellent structural preservation and visual fidelity, robust even in complex abdomen-to-MR translation.
- **Open-source code**: ✅ [GitHub Repo](https://github.com/Mors20/Structure-Preserving-Unpaired-Medical-Image-Translation)
- **Ideas to Borrow**:
    - ISR mechanism for balancing structural conditioning vs. realism
    - Edge reshaping using attention maps + Gaussian blur + re-noising
    - Conditional + unconditional training phases for flexible sampling

## SurgicaL-CD: Generating Surgical Images via Unpaired Image Translation with Latent Consistency Diffusion Models (2024)

**Authors**: Danush Kumar Venkatesh, Dominik Rivoir, Micha Pfeiffer, Stefanie Speidel  
**Venue**: arxiv
**Link**: [https://arxiv.org/abs/2408.09822](https://arxiv.org/abs/2408.09822)

### 🧠 Method

- **Key Technique**: Latent Consistency Distillation (LCD) applied to diffusion models
- **Type**: Unpaired
- **Architecture Notes**: Utilizes Stable Diffusion fine-tuned on surgical datasets, followed by consistency distillation to enable few-step image generation. Incorporates SDEdit for image translation and optionally employs ControlNet for spatial conditioning.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Simulated surgical images
- **Target Modality**: Realistic surgical images
- **Paired/Unpaired Data**: No (Unpaired)

### 🌟 Highlights

- Generates high-quality, realistic surgical images from simulations without paired data
- Achieves image synthesis in fewer than five sampling steps
- Outperforms GAN-based and previous diffusion approaches in image quality and diversity
- Demonstrates utility of generated images in downstream surgical applications, such as segmentation tasks
- Reduces annotation costs by leveraging automatically rendered labels from simulations

### 📊 Evaluation

- **Metrics Used**:
    - Image quality assessments
    - Semantic consistency evaluations
    - Performance in downstream segmentation tasks
- **Datasets**:
    - CholecSeg8K (real surgical images)
    - CholecT50 (real surgical images)
    - Dresden Surgical Anatomy Dataset (DSAD) (real surgical images)
    - Lap and Gast simulated surgical scenes
- **Baseline Comparison**:
    - GAN-based unpaired image translation methods
    - Previous diffusion-based approaches

### 📌 Summary Notes

- **Strengths**: Efficient generation of realistic surgical images with corresponding labels, enhancing machine learning model training for surgical applications.
- **Open-source code**: ✅ [GitLab Repository](https://gitlab.com/nct_tso_public/gan2diffusion)
- **Ideas to Borrow**:
    - Application of latent consistency distillation to reduce sampling steps in diffusion models
    - Integration of SDEdit for unpaired image translation tasks
    - Use of ControlNet for spatial conditioning to preserve anatomical structures during image synthesis

# 1.3 Diffusion-Based Methods

## DCE-diff: Diffusion Model for Synthesis of Early and Late Dynamic Contrast-Enhanced MR Images from Non-Contrast Multimodal Inputs (2024)  
**Authors**: Kishore Kumar M, Sriprabha Ramanarayanan, Sadhana S, Arunima Sarkar, Matcha Naga Gayathri, Keerthi Ram, Mohanasankar Sivaprakasam  
**Venue**: CVPR Workshops 2024  
**Link**: [https://doi.org/10.1109/CVPRW63382.2024.00525](https://doi.org/10.1109/CVPRW63382.2024.00525)

### 🧠 Method
- **Key Technique**: Multimodal conditional diffusion model
- **Type**: Paired
- **Architecture Notes**: U-Net based denoising network with residual layers, self-attention, and time-step embeddings; diffusion model trained to generate both early and late DCE-MRI images from T2, PD, T1 pre-contrast, and ADC maps

### 🔄 Input ↔ Output Modalities
- **Source Modality**: Multimodal MRI: T2W, PD, T1-precontrast, ADC
- **Target Modality**: DCE-MRI (early- and late-phase)
- **Paired/Unpaired Data**: Paired

### 🌟 Highlights
- Avoids use of Gadolinium contrast agents for DCE-MRI
- Learns from multimodal structural + perfusion images (ADC)
- Outperforms state-of-the-art GAN and transformer-based methods (Pix2Pix, RegGAN, ResViT, TSGAN, ConvLSTM)
- Highly robust to domain shift (generalizes from ProstateX to Prostate-MRI)
- Ablation study confirms the importance of ADC input in quality of DCE synthesis

### 📊 Evaluation
- **Metrics Used**: PSNR, SSIM, MAE, FID
- **Datasets**:  
  - **ProstateX** (training & in-domain test)  
  - **Prostate-MRI** (cross-domain test)
- **Baseline Comparison**: Pix2Pix, RegGAN, TSGAN, ResViT, ConvLSTM

### 📌 Summary Notes
- **Strengths**: Dual-phase contrast image synthesis from non-contrast multimodal input; clinically applicable; superior generalization to unseen scanner settings
- **Open-source code**: ✖️ Not provided
- **Ideas to Borrow**: Use of ADC as perfusion conditioning; dual target synthesis in diffusion training; domain-generalization evaluation; combining multiple structural modalities as conditioning signal

## Robust Cross-modal Medical Image Translation via Diffusion Model and Knowledge Distillation (2024)  
**Authors**: Yuehan Xia, Saifeng Feng, Jianhui Zhao, Zhiyong Yuan  
**Venue**: IJCNN 2024 (International Joint Conference on Neural Networks)  
**Link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/10650498)

### 🧠 Method
- **Key Technique**: Combines GAN with forward diffusion and a teacher-student knowledge distillation refinement network
- **Type**: Paired and Unpaired (evaluated on both)
- **Architecture Notes**: 
  - Generator followed by forward diffusion for noise injection  
  - Time-dependent discriminator trained on noisy versions of real/generated images  
  - Knowledge distillation uses a teacher registration network (with U-Net + STN) and a structurally similar student module

### 🔄 Input ↔ Output Modalities
- **Source Modality**: T1 MRI
- **Target Modality**: T2 MRI
- **Paired/Unpaired Data**: Both paired and unpaired data are used (e.g., BraTS, BrainWeb)

### 🌟 Highlights
- Uses forward diffusion as data augmentation in adversarial training (without reverse denoising)
- Improves robustness via Gaussian mixture noise modeling at various diffusion steps
- Student refinement module enables high-quality results even without the target image during testing
- Outperforms GAN and diffusion-based baselines under various noise and misalignment perturbations

### 📊 Evaluation
- **Metrics Used**: MAE, PSNR, SSIM
- **Datasets**: BraTS 2018, BrainWeb (both T1-T2 MR)
- **Baseline Comparison**: Pix2Pix, CycleGAN, UNIT, MUNIT, RegGAN, DFMIR, Swin, SynDiff

### 📌 Summary Notes
- **Strengths**: High robustness to noise and misalignment, simple training, no reverse diffusion needed
- **Open-source code**: ✖️ Not yet available
- **Ideas to Borrow**:
  - Forward-only diffusion module for noise injection during GAN training
  - Knowledge distillation with attention map transfer for unsupervised refinement
  - Use of time-dependent discriminator trained on multiple noise levels

## Self-Consistent Recursive Diffusion Bridge for Medical Image Translation (2024)

**Authors**: Fuat Arslan, Bilal Kabas, Onat Dalmaz, Muzaffer Ozbey, Tolga Çukur  
**Venue**: arXiv (preprint)  
**Link**: [https://arxiv.org/abs/2405.06789](https://arxiv.org/abs/2405.06789)  
**Code**: [https://github.com/icon-lab/SelfRDB](https://github.com/icon-lab/SelfRDB)

### 🧠 Method

- **Key Technique**: Self-Consistent Recursive Diffusion Bridge (SelfRDB)
- **Type**: Paired
- **Architecture Notes**: Diffusion bridge with a novel forward process using a **soft-prior** on the source image and a recursive generator in reverse steps for **self-consistent target estimation**. Incorporates adversarial training with a discriminator.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: CT, T2w, PDw, FLAIR
- **Target Modality**: MRI (T1w)
- **Paired/Unpaired Data**: Yes (Paired)

### 🌟 Highlights

- First diffusion bridge specifically for medical image translation
- **Monotonically increasing noise schedule** enables better generalization from source modality
- Reverse process uses **recursive estimation** of target image until convergence
- Outperforms DDPM, SynDiff, I2SB, and pix2pix across multiple MRI and CT tasks
- Includes detailed ablation study proving the value of each component

### 📊 Evaluation

- **Metrics Used**: PSNR, SSIM
- **Datasets**:
    - IXI (T1/T2/PD brain MR)
    - BRATS (T1/T2/FLAIR glioma MR)
    - Pelvic MRI-CT (Gold Atlas)
- **Baseline Comparison**: DDPM, SynDiff, I2SB (diffusion bridge), pix2pix

### 📌 Summary Notes

- **Strengths**: Accurate anatomy preservation, high-fidelity synthesis, robust to source variability
- **Open-source code**: ✅ [GitHub Repo](https://github.com/icon-lab/SelfRDB)
- **Ideas to Borrow**: Soft-prior guided noise schedule, recursive self-consistency in reverse sampling, adversarial loss integrated in diffusion bridge

## Mutual Information Guided Diffusion for Zero-Shot Cross-Modality Medical Image Translation (2024)

**Authors**: Zihao Wang, Yingyu Yang, Yuzhou Chen, Tingting Yuan, Maxime Sermesant, Hervé Delingette, Ona Wu  
**Venue**: IEEE Transactions on Medical Imaging (TMI)  
**Link**: [https://doi.org/10.1109/TMI.2024.3382043](https://doi.org/10.1109/TMI.2024.3382043)  
**Code**: [https://github.com/mgh-ccni/midiffusion](https://github.com/mgh-ccni/midiffusion)

### 🧠 Method

- **Key Technique**: Mutual Information-Guided Diffusion Model (MIDiffusion)
- **Type**: Zero-shot, Unsupervised
- **Architecture Notes**: Score-based diffusion model guided by Local-wise Mutual Information (LMI); includes differentiable LMI layer for conditioning denoising steps; uses SDE (variance exploding)

### 🔄 Input ↔ Output Modalities

- **Source Modality**: CT, FLAIR, PDw, T2w, etc.
- **Target Modality**: T1w (MRI)
- **Paired/Unpaired Data**: No (Zero-shot: only target domain seen during training)

### 🌟 Highlights

- Requires no source domain training data
- LMI guidance enables statistically consistent, faithful image translation
- Robust in 2D and 3D translation settings (used for segmentation)
- Outperforms SDEdit, CycleGAN, EGSDE, and StyleGAN inversion in most cases
- Introduces CSLMI metric to assess modality translatability

### 📊 Evaluation

- **Metrics Used**: SSIM, GMSD, PSNR, MSE, FID, Mutual Information
- **Datasets**: GoldAtlas (CT/MR), CuRIOUS (T1w/FLAIR), IXI (T1w/PDw)
- **Baseline Comparison**: CycleGAN, StyleGAN2-ADA (inversion), SDEdit (t=0.2/0.5), EGSDE

### 📌 Summary Notes

- **Strengths**: Best fidelity in unsupervised zero-shot modality translation; strong anatomical preservation; applicable in downstream 3D segmentation
- **Open-source code**: ✅ [GitHub Repo](https://github.com/mgh-ccni/midiffusion)
- **Ideas to Borrow**: Statistical guidance with local MI, CSLMI for modality translatability, LMI-based diffusion conditioning for zero-shot translation

## A Diffusion Model Translator for Efficient Image-to-Image Translation (2024)

**Authors**: Mengfei Xia, Yu Zhou, Ran Yi, Yong-Jin Liu, Wenping Wang  
**Venue**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)  
**Link**: [DOI:10.1109/TPAMI.2024.3435448](https://doi.org/10.1109/TPAMI.2024.3435448)

### 🧠 Method

- **Key Technique**: Diffusion Model Translator (DMT) with shared noise encoding and intermediate timestep domain transfer
- **Type**: Paired
- **Architecture Notes**: Combines pretrained DDPM and a lightweight translator module (e.g., Pix2Pix/TSIT-based), optimized for a single intermediate timestep rather than full-step conditioning

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Varies by task (e.g., sketch, segmentation mask, grayscale image)
- **Target Modality**: Stylized image, photo, colorized image, etc.
- **Paired/Unpaired Data**: Yes (Paired data used in training)

### 🌟 Highlights

- Dramatically improves efficiency by applying translation only at a single timestep instead of every step
- Theoretically justifies the sufficiency of midpoint domain translation
- Automatically selects the best timestep via SSIM-based strategy
- Supports asymmetric and multi-step translation extensions
- 40×–80× faster than Palette and higher quality than GAN baselines

### 📊 Evaluation

- **Metrics Used**: FID, SSIM, LPIPS, L1, L2, user study (1–5 scores)
- **Datasets**:
    - Portrait stylization: CelebA-HQ (via QMUPD)
    - AFHQ (animal images)
    - CelebA-HQ with segmentation
    - Edges2Handbags (sketch-to-image)
- **Baseline Comparison**:
    - GAN-based: Pix2Pix, TSIT, SPADE, QMUPD
    - Diffusion-based: Palette

### 📌 Summary Notes

- **Strengths**: High-quality outputs, fast training and inference, theoretically grounded design
- **Open-source code**: ✅ Yes ([GitHub Repo](https://github.com/THU-LYJ-Lab/dmt))
- **Ideas to Borrow**:
    - Shared encoder with intermediate domain transfer
    - Timestep optimization via SSIM intersection strategy
    - Lightweight translator detached from diffusion model retraining

Let me know if you want all three papers compared in a table or included in your literature review mind map!

## Zero-shot Medical Image Translation via Frequency-Guided Diffusion Models (2023)

**Authors**: Yunxiang Li, Hua-Chieh Shao, Xiao Liang, Liyuan Chen, Ruiqi Li, Steve Jiang, Jing Wang, You Zhang  
**Venue**: arXiv  
**Link**: [https://arxiv.org/abs/2304.02742](https://arxiv.org/abs/2304.02742)

### 🧠 Method

- **Key Technique**: Frequency-Guided Diffusion Model (FGDM)
- **Type**: Unpaired
- **Architecture Notes**: Employs frequency-domain filters to guide the diffusion process, preserving structural details during image translation.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Cone-beam CT (CBCT)
- **Target Modality**: CT
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Addresses the challenge of structural information loss in diffusion models during medical image translation.
- Demonstrates significant improvements over GAN-based, VAE-based, and other diffusion-based methods in zero-shot medical image translation tasks.

### 📊 Evaluation

- **Metrics Used**: Fréchet Inception Distance (FID), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM)
- **Datasets**: CBCT-to-CT translation tasks across different anatomical sites
- **Baseline Comparison**: Outperforms state-of-the-art methods in FID, PSNR, and SSIM metrics.

### 📌 Summary Notes

- Highlights the effectiveness of frequency-domain guidance in preserving anatomical structures during medical image translation.
- Demonstrates the potential of zero-shot learning in medical imaging applications.

## Cross-conditioned Diffusion Model for Medical Image to Image Translation (2024)

**Authors**: Zhaohu Xing, Sicheng Yang, Sixiang Chen, Tian Ye, Yijun Yang, Jing Qin, Lei Zhu  
**Venue**: arXiv  
**Link**: [https://arxiv.org/abs/2409.08500](https://arxiv.org/abs/2409.08500)

### 🧠 Method

- **Key Technique**: Cross-conditioned Diffusion Model (CDM)
- **Type**: Unpaired
- **Architecture Notes**: Utilizes the distribution of target modalities as guidance to improve synthesis quality and achieve higher generation efficiency.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various medical imaging modalities
- **Target Modality**: Corresponding translated modalities
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Addresses the complexity and computational expense associated with training GANs and diffusion models for medical image translation.
- Demonstrates improved performance in synthesis quality and generation efficiency compared to conventional diffusion models.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: -
- **Baseline Comparison**: Compared against traditional GAN and diffusion models

### 📌 Summary Notes

- Introduces a novel approach to medical image translation by leveraging cross-conditioned diffusion models.
- Emphasizes the importance of target modality distribution guidance in improving synthesis quality.

## Cascaded Multi-path Shortcut Diffusion Model for Medical Image Translation (2024)

**Authors**: Yinchi Zhou, Tianqi Chen, Jun Hou, Huidong Xie, Nicha C. Dvornek, S. Kevin Zhou, David L. Wilson, James S. Duncan, Chi Liu, Bo Zhou  
**Venue**: Medical Image Analysis  
**Link**: [https://www.sciencedirect.com/science/article/abs/pii/S1361841524002251](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002251)

### 🧠 Method

- **Key Technique**: Cascaded Multi-path Shortcut Diffusion Model (CMDM)
- **Type**: Unpaired
- **Architecture Notes**: Combines GAN and diffusion models, incorporating a multi-path shortcut diffusion strategy and a cascaded pipeline for enhanced translation quality and uncertainty estimation.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various medical imaging modalities
- **Target Modality**: Corresponding translated modalities
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Addresses the instability and lack of uncertainty estimation in GAN-based methods.
- Demonstrates high-quality translations with robust performance and reasonable uncertainty estimations.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Three different medical image datasets with two sub-tasks each
- **Baseline Comparison**: Compared against state-of-the-art methods

### 📌 Summary Notes

- Proposes a novel integration of GAN and diffusion models for medical image translation.
- Emphasizes the importance of uncertainty estimation in medical imaging applications.

## Unsupervised Medical Image Translation with Adversarial Diffusion Models (2022)

**Authors**: Muzaffer Özbey, Onat Dalmaz, Salman UH Dar, Hasan A Bedel, Şaban Özturk, Alper Güngör, Tolga Çukur  
**Venue**: IEEE Transactions on Medical Imaging  
**Link**: [https://ieeexplore.ieee.org/document/10167641/authors#authors](https://ieeexplore.ieee.org/document/10167641/authors#authors)

### 🧠 Method

- **Key Technique**: Adversarial Diffusion Modeling (SynDiff)
- **Type**: Unpaired
- **Architecture Notes**: Leverages a conditional diffusion process with adversarial projections in the reverse diffusion direction, incorporating a cycle-consistent architecture for unpaired datasets.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various medical imaging modalities
- **Target Modality**: Corresponding translated modalities
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Addresses the limitations of GAN models in sample fidelity for medical image translation.
- Demonstrates superior performance in multi-contrast MRI and MRI-CT translation tasks.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Multi-contrast MRI and MRI-CT datasets
- **Baseline Comparison**: Compared against competing GAN and diffusion models

### 📌 Summary Notes

- Introduces adversarial diffusion modeling as a novel approach for medical image translation.
- Emphasizes the importance of capturing direct correlates of image distribution for improved performance.

# 1.4 Multi-Modal Fusion Translation

## MedFusionGAN: Multimodal Medical Image Fusion Using an Adaptive Weighted Fusion Strategy (2022)

**Authors**: Mojtaba Safari, Ali Fatemi, Louis Archambault
**Venue**: BMC Medical Imaging  
**Link**: [https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-023-01160-w](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-023-01160-w)

### 🧠 Method

- **Key Technique**: Generative Adversarial Network (GAN) with adaptive weighted fusion strategy
- **Type**: Paired
- **Architecture Notes**: Implements a GAN framework that adaptively combines MRI soft-tissue contrast and CT bone information into a single fused image.

### 🔄 Input ↔ Output Modalities

- **Source Modalities**: MRI and CT images
- **Target Modality**: Fused MRI-CT image
- **Paired/Unpaired Data**: Paired

### 🌟 Highlights

- Successfully generates fused images that retain both MRI soft-tissue details and CT bone contrast.
- Outperforms seven traditional and eight deep learning-based state-of-the-art methods in preserving spatial resolution without introducing artifacts.
- Demonstrates qualitative improvements in distinguishing boundaries between bone and scalp, as well as between white and gray matter.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: MRI and CT images from clinical studies
- **Baseline Comparison**: Compared against traditional fusion methods like FPDE, GTF, GFDG, IVF, MEF, and deep learning methods including FusionGAN, SESF-Fuse, CNN-Fuse, and U2Fusion.

### 📌 Summary Notes

- Introduces an adaptive weighted fusion strategy within a GAN framework for effective multimodal medical image fusion.
- Emphasizes the importance of preserving both spatial resolution and contrast information from source modalities without introducing artifacts.

## Multimodal Image Fusion: A Systematic Review (2023)

**Authors**: Shrida Kalamkar , Geetha Mary A.
**Venue**: Decision Analytics Journal
**Link**: [https://www.sciencedirect.com/science/article/pii/S2772662223001674](https://www.sciencedirect.com/science/article/pii/S2772662223001674)

### 🧠 Method

- **Key Technique**: Systematic review of multimodal image fusion techniques
- **Type**: Review
- **Architecture Notes**: Analyzes various methods and approaches used in multimodal image fusion, categorizing them based on their methodologies and applications.

### 🔄 Input ↔ Output Modalities

- **Source Modalities**: Various imaging modalities (e.g., MRI, CT, PET)
- **Target Modality**: Fused images
- **Paired/Unpaired Data**: Both paired and unpaired

### 🌟 Highlights

- Provides a comprehensive overview of existing multimodal image fusion techniques.
- Discusses the advantages and limitations of different fusion methods.
- Highlights the importance of multimodal fusion in enhancing diagnostic accuracy and clinical decision-making.

### 📊 Evaluation

- **Metrics Used**: Not applicable
- **Datasets**: Various datasets from reviewed studies
- **Baseline Comparison**: Not applicable

### 📌 Summary Notes

- Serves as a valuable resource for researchers and practitioners interested in multimodal image fusion.
- Emphasizes the need for developing advanced fusion techniques that can effectively integrate information from multiple imaging modalities.

## A Brief Analysis of Multimodal Medical Image Fusion Techniques (2023)

**Authors**: Mohammed Ali Saleh, AbdElmgeid A. Ali, Kareem Ahmed and Abeer M. Sarhan
**Venue**: MDPI Electronics  
**Link**: [https://www.mdpi.com/2079-9292/12/1/97](https://www.mdpi.com/2079-9292/12/1/97)

### 🧠 Method

- **Key Technique**: Review of multimodal medical image fusion techniques
- **Type**: Review
- **Architecture Notes**: Discusses various fusion techniques, including traditional and deep learning-based methods, highlighting their applications and performance.

### 🔄 Input ↔ Output Modalities

- **Source Modalities**: Various medical imaging modalities
- **Target Modality**: Fused images
- **Paired/Unpaired Data**: Both paired and unpaired

### 🌟 Highlights

- Analyzes the effectiveness of different fusion techniques in enhancing the quality of medical images.
- Emphasizes the role of multimodal fusion in improving diagnostic accuracy.
- Discusses the challenges and future directions in the field of medical image fusion.

### 📊 Evaluation

- **Metrics Used**: Not applicable
- **Datasets**: Various datasets from reviewed studies
- **Baseline Comparison**: Not applicable

### 📌 Summary Notes

- Provides insights into the current state of multimodal medical image fusion techniques.
- Highlights the need for developing more robust and efficient fusion methods to meet clinical requirements.

## Enhanced Multimodal Medical Image Fusion via Modified DWT with Arithmetic Optimization Algorithm (2023)

**Authors**: Ahmad A. Alzahrani
**Venue**: Scientific Reports  
**Link**: [https://www.nature.com/articles/s41598-024-69997-x](https://www.nature.com/articles/s41598-024-69997-x)

### 🧠 Method

- **Key Technique**: Modified Discrete Wavelet Transform (DWT) with Arithmetic Optimization Algorithm (AOA)
- **Type**: Paired
- **Architecture Notes**: Utilizes a modified DWT approach combined with AOA to fuse multimodal medical images, aiming to retain significant details and features from each modality.

### 🔄 Input ↔ Output Modalities

- **Source Modalities**: Various medical imaging modalities
- **Target Modality**: Fused images
- **Paired/Unpaired Data**: Paired

### 🌟 Highlights

- Enhances diagnostic accuracy by combining complementary information from different imaging modalities.
- Demonstrates improved performance over traditional fusion methods in preserving edge details and reducing artifacts.
- Incorporates bilateral filtering for noise elimination prior to fusion.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: -
- **Baseline Comparison**: Compared against traditional fusion methods

### 📌 Summary Notes

- Introduces a novel fusion model that combines modified DWT with AOA for effective multimodal medical image fusion.
- Emphasizes the importance of noise elimination and detail preservation in the fusion process.

# 1.5 Weakly-Supervised / Self-Supervised Learning

## Transferable Visual Words: Exploiting the Semantics of Anatomical Patterns for Self-supervised Learning (2021)

**Authors**: Fatemeh Haghighi, Mohammad Reza Hosseinzadeh Taher, Zongwei Zhou, Michael B. Gotway, Jianming Liang  
**Venue**: arXiv  
**Link**: [https://arxiv.org/abs/2102.10680](https://arxiv.org/abs/2102.10680)

### 🧠 Method

- **Key Technique**: Transferable Visual Words (TransVW) for self-supervised learning
- **Type**: Self-supervised
- **Architecture Notes**: Introduces a method to automatically discover anatomical patterns ("visual words") in medical images, which are then used to pre-train models in a self-supervised manner.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various medical imaging modalities
- **Target Modality**: N/A (focus on representation learning)
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Demonstrates that models pre-trained using TransVW require fewer annotated samples for downstream tasks.
- Achieves higher performance and faster convergence compared to models trained from scratch or with other self-supervised methods.

### 📊 Evaluation

- **Metrics Used**: Performance on downstream tasks (e.g., classification, segmentation)
- **Datasets**: Publicly available medical imaging datasets
- **Baseline Comparison**: Compared against models trained from scratch and other self-supervised learning approaches

### 📌 Summary Notes

- Introduces a novel approach to self-supervised learning by leveraging anatomical patterns inherent in medical images.
- Emphasizes the potential of reducing annotation costs while maintaining high performance in medical image analysis tasks.

## Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis (2021)

**Authors**: Yucheng Tang, Dong Yang, Wenqi Li, Holger Roth, Bennett Landman, Daguang Xu, Vishwesh Nath, Ali Hatamizadeh  
**Venue**: arXiv  
**Link**: [https://arxiv.org/abs/2111.14791](https://arxiv.org/abs/2111.14791)

### 🧠 Method

- **Key Technique**: Self-supervised pre-training of Swin Transformers
- **Type**: Self-supervised
- **Architecture Notes**: Proposes a 3D transformer-based model, Swin UNETR, with a hierarchical encoder pre-trained using tailored proxy tasks to learn anatomical structures from unlabeled 3D medical images.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: 3D medical images (e.g., CT scans)
- **Target Modality**: N/A (focus on representation learning)
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Achieves state-of-the-art performance on multiple medical image segmentation tasks after fine-tuning.
- Demonstrates the effectiveness of transformer-based architectures in capturing both global and local contextual information in medical images.

### 📊 Evaluation

- **Metrics Used**: Dice Similarity Coefficient (DSC), Hausdorff Distance
- **Datasets**: Beyond the Cranial Vault (BTCV) Segmentation Challenge, Medical Segmentation Decathlon (MSD)
- **Baseline Comparison**: Outperforms previous state-of-the-art models on the BTCV and MSD datasets.

### 📌 Summary Notes

- Highlights the potential of self-supervised learning in pre-training large-scale transformer models for medical image analysis.
- Emphasizes the importance of learning rich representations from unlabeled data to improve performance on downstream tasks with limited annotations.

# 1.6 Task-Guided Translation

## Uncertainty-Guided Progressive GANs for Medical Image Translation (2021)

**Authors**: Uddeshya Upadhyay, Yanbei Chen, Tobias Hepp, Sergios Gatidis, Zeynep Akata  
**Venue**: International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)  
**Link**: [https://link.springer.com/chapter/10.1007/978-3-030-87199-4_58](https://link.springer.com/chapter/10.1007/978-3-030-87199-4_58)

### 🧠 Method

- **Key Technique**: Uncertainty-Guided Progressive Learning in Generative Adversarial Networks (GANs)
- **Type**: Unpaired
- **Architecture Notes**: Incorporates aleatoric uncertainty as attention maps within a progressive GAN framework to focus on regions with higher uncertainty during training, thereby improving image translation quality.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: PET, undersampled MRI, motion-corrupted MRI
- **Target Modality**: CT, fully-sampled MRI, motion-corrected MRI
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Addresses the challenge of uncertainty estimation in medical image translation.
- Demonstrates improved performance across multiple tasks, including PET to CT translation, undersampled MRI reconstruction, and MRI motion artifact correction.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Various datasets corresponding to each translation task
- **Baseline Comparison**: Compared against state-of-the-art GAN-based methods

### 📌 Summary Notes

- Highlights the importance of incorporating uncertainty estimation into GAN-based medical image translation models.
- Suggests that focusing on uncertain regions during training can lead to higher fidelity translations.

## TarGAN: Target-Aware Generative Adversarial Networks for Multi-modality Medical Image Translation (2021)

**Authors**: Junxiao Chen, Jia Wei, Rui Li  
**Venue**: arXiv  
**Link**: [https://arxiv.org/abs/2105.08993](https://arxiv.org/abs/2105.08993)

### 🧠 Method

- **Key Technique**: Target-Aware Generative Adversarial Network (TarGAN)
- **Type**: Unpaired
- **Architecture Notes**: Jointly learns whole image translation and target area translation mappings, with a crossing loss to interrelate these mappings, enhancing the quality of the translated target area.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various medical imaging modalities
- **Target Modality**: Corresponding translated modalities
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Focuses on improving the quality of target areas or regions of interest (ROIs) in translated images.
- Demonstrates superior performance in translating critical regions compared to existing methods.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: -
- **Baseline Comparison**: Compared against state-of-the-art unpaired image translation methods

### 📌 Summary Notes

- Emphasizes the importance of accurately translating target areas in medical images.
- Introduces a novel approach to jointly learn whole image and target area translations.

## Discriminative Cross-Modal Data Augmentation for Medical Imaging Applications (2020)

**Authors**: Yue Yang, Pengtao Xie  
**Venue**: arXiv  
**Link**: [https://arxiv.org/abs/2010.03468](https://arxiv.org/abs/2010.03468)

### 🧠 Method

- **Key Technique**: Discriminative Unpaired Image-to-Image Translation
- **Type**: Unpaired
- **Architecture Notes**: Integrates the translation task with a downstream prediction task, where the translation is guided by the prediction, enhancing the quality of the translated images for the specific task.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various medical imaging modalities
- **Target Modality**: Corresponding translated modalities
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Addresses the challenge of limited labeled medical images by augmenting data through cross-modal translation.
- Demonstrates improved performance in downstream prediction tasks by integrating task-specific guidance into the translation process.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Two applications demonstrating the effectiveness of the method
- **Baseline Comparison**: Compared against standard unpaired image-to-image translation models

### 📌 Summary Notes

- Highlights the potential of integrating downstream tasks into the image translation process to enhance performance.
- Suggests that task-guided translation can effectively augment data for medical imaging applications.

# 2. Natural Image Translation (Inspiration & Foundation)

# 2.1 Classic Image-to-Image GANs

## Image-to-Image Translation with Conditional Adversarial Networks (2017)

**Authors**: Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros  
**Venue**: IEEE Conference on Computer Vision and Pattern Recognition (CVPR)  
**Link**: [https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf)

### 🧠 Method

- **Key Technique**: Conditional Generative Adversarial Network (cGAN)
- **Type**: Paired
- **Architecture Notes**: Utilizes a U-Net-based generator and a PatchGAN discriminator to learn a mapping from input to output images conditioned on paired data.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various (e.g., sketches, segmentation maps)
- **Target Modality**: Corresponding realistic images
- **Paired/Unpaired Data**: Paired

### 🌟 Highlights

- Demonstrates the versatility of cGANs in performing various image-to-image translation tasks.
- Emphasizes the importance of paired datasets in achieving high-quality translations.

### 📊 Evaluation

- **Metrics Used**: Mean Squared Error (MSE), Structural Similarity Index (SSIM)
- **Datasets**: Facades, Cityscapes, and others
- **Baseline Comparison**: Compared against traditional methods for each specific task

### 📌 Summary Notes

- Introduces the pix2pix framework, which has become foundational in paired image translation tasks.
- Provides a general-purpose solution applicable to various domains, including medical imaging.

## Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (2017)

**Authors**: Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros  
**Venue**: IEEE International Conference on Computer Vision (ICCV)  
**Link**: [https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)

### 🧠 Method

- **Key Technique**: Cycle-Consistent Generative Adversarial Network (CycleGAN)
- **Type**: Unpaired
- **Architecture Notes**: Employs two generator-discriminator pairs to learn mappings between two domains without paired examples, enforcing cycle consistency to ensure meaningful translations.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various (e.g., photographs)
- **Target Modality**: Corresponding translated images (e.g., paintings)
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Enables image-to-image translation without the need for paired datasets.
- Introduces cycle consistency loss to enforce inverse mappings, ensuring that translating an image to another domain and back results in the original image.

### 📊 Evaluation

- **Metrics Used**: Qualitative assessments, user studies
- **Datasets**: Collection of images from different domains (e.g., horses↔zebras, summer↔winter)
- **Baseline Comparison**: Compared against traditional methods and ablated versions of the model

### 📌 Summary Notes

- Presents a breakthrough in unpaired image-to-image translation, allowing for flexible domain adaptation.
- The CycleGAN framework has been widely adopted and extended in various applications, including medical image analysis.

# 2.2 Disentanglement & Style-Based Translation

## Multi-mapping Image-to-Image Translation via Learning Disentanglement (2019)

**Authors**: Xiaoming Yu, Yuanqi Chen, Shan Liu, Thomas Li, Ge Li
**Venue**: Advances in Neural Information Processing Systems (NeurIPS)  
**Link**: [https://papers.nips.cc/paper/8564-multi-mapping-image-to-image-translation-via-learning-disentanglement](https://papers.nips.cc/paper/8564-multi-mapping-image-to-image-translation-via-learning-disentanglement)

### 🧠 Method

- **Key Technique**: Disentangled representation learning for multi-mapping image translation
- **Type**: Unpaired
- **Architecture Notes**: Introduces a framework that learns disentangled content and style representations, enabling multi-modal and multi-domain image translations using a single unified model.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various image domains
- **Target Modality**: Corresponding translated images in target domains
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Achieves multi-mapping translation by learning separate content and style representations.
- Enables both multi-modal (diverse outputs within a target domain) and multi-domain (translations across different domains) capabilities.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Various datasets corresponding to different translation tasks
- **Baseline Comparison**: Compared against state-of-the-art image-to-image translation methods

### 📌 Summary Notes

- Proposes a unified framework that addresses both multi-modal and multi-domain translation tasks through disentanglement.
- Demonstrates the effectiveness of learning aligned style representations across domains for improved translation quality.

## Image-to-Image Translation via Hierarchical Style Disentanglement (2021)

**Authors**: Xinyang Li, Shengchuan Zhang, Jie Hu, Liujuan Cao, Xiaopeng Hong, Xudong Mao, Feiyue Huang, Yongjian Wu, Rongrong Ji  
**Venue**: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)  
**Link**: [https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Image-to-Image_Translation_via_Hierarchical_Style_Disentanglement_CVPR_2021_paper.pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Image-to-Image_Translation_via_Hierarchical_Style_Disentanglement_CVPR_2021_paper.pdf)

### 🧠 Method

- **Key Technique**: Hierarchical Style Disentanglement (HiSD)
- **Type**: Unpaired
- **Architecture Notes**: Organizes labels into a hierarchical structure, allocating independent tags, exclusive attributes, and disentangled styles, facilitating controllable and diverse translations.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various image domains
- **Target Modality**: Corresponding translated images with specific attributes
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Addresses multi-label and multi-style translation challenges by hierarchically disentangling styles.
- Enables controllable translations by manipulating specific attributes while preserving others.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: CelebA-HQ dataset
- **Baseline Comparison**: Compared against existing multi-label and multi-style translation methods

### 📌 Summary Notes

- Introduces a hierarchical approach to style disentanglement, enhancing control over translated attributes.
- Demonstrates improved visual quality and diversity in translated images.

## Diagonal Attention and Style-based GAN for Content-Style Disentanglement in Image Generation and Translation (2021)

**Authors**: Gihyun Kwon, Jong Chul Ye  
**Venue**: arXiv  
**Link**: [https://arxiv.org/abs/2103.16146](https://arxiv.org/abs/2103.16146)

### 🧠 Method

- **Key Technique**: Diagonal Attention (DAT) and Style-based Generative Adversarial Network (GAN)
- **Type**: Unpaired
- **Architecture Notes**: Introduces DAT layers to separately manipulate spatial contents from styles hierarchically, enabling coarse-to-fine level disentanglement.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various image domains
- **Target Modality**: Corresponding translated images with disentangled content and style
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Achieves improved content-style disentanglement by integrating DAT with style-based GANs.
- Enables flexible control over spatial features in generated images.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Various datasets corresponding to different translation tasks
- **Baseline Comparison**: Compared against existing content-style disentanglement methods

### 📌 Summary Notes

- Proposes a novel attention mechanism to enhance content-style disentanglement in image translation.
- Demonstrates superior performance in controlling spatial content and style attributes.

# 2.3 Conditioning & Structure Control

## Contrastive Learning Guided Latent Diffusion Model for Image-to-Image Translation (2024)

**Authors**: Qi Si, Bo Wang, Zhao Zhang, Mingbo Zhao, Xiaojie Jin, Mingliang Xu, Meng Wang  
**Venue**: arXiv
**Link**: [https://arxiv.org/abs/2503.20484](https://arxiv.org/abs/2503.20484)

### 🧠 Method

- **Key Technique**: Zero-shot latent diffusion (pix2pix-zeroCon) + contrastive loss (CUT) + automatic prompt generation
- **Type**: Training-free / Zero-shot
- **Architecture Notes**: Latent Diffusion Model (Stable Diffusion v1.5 backbone) guided by:
    - Cross-Attention Map Loss
    - Patch-wise CUT Loss
    - Editing direction in text embedding space from BLIP + CLIP

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Natural image (e.g., animal, face, scene)
- **Target Modality**: Edited version based on text (e.g., add glasses, change season, change species)
- **Paired/Unpaired Data**: No (Zero-shot, unpaired)

### 🌟 Highlights

- Training-free (no model finetuning)
- BLIP and CLIP used to auto-generate prompts and editing directions
- Strong structure and content preservation using combined losses
- Effective in many editing types: face change, object editing, scene transformation
- Competitive results on real-world benchmarks (LAION-5B, CelebAMask-HQ, ImageNet-R)

### 📊 Evaluation

- **Metrics Used**: CLIP-Score (text-image alignment), BG-LPIPS (background content preservation), DINO-ViT Structure Distance
- **Datasets**: CelebAMask-HQ-512, ImageNet-R, LAION-5B
- **Baseline Comparison**: DDIM + word swap, pnp, p2p-zero, CDS, DDS

### 📌 Summary Notes

- **Strengths**: Strong zero-shot editing with automatic prompt generation, combines CUT + attention loss for best structure + content preservation
- **Open-source code**: ✖️ (Not mentioned yet)
- **Ideas to Borrow**: Multi-sentence CLIP-based direction computation, attention gradient descent to guide image editing, joint cross-attention and contrastive loss in LDM inference

## Guided Image-to-Image Translation with Bi-Directional Feature Transformation (2019)

**Authors**: Badour AlBahar, Jia-Bin Huang  
**Venue**: IEEE International Conference on Computer Vision (ICCV)  
**Link**: [https://openaccess.thecvf.com/content_ICCV_2019/papers/AlBahar_Guided_Image-to-Image_Translation_With_Bi-Directional_Feature_Transformation_ICCV_2019_paper.pdf](https://openaccess.thecvf.com/content_ICCV_2019/papers/AlBahar_Guided_Image-to-Image_Translation_With_Bi-Directional_Feature_Transformation_ICCV_2019_paper.pdf)

### 🧠 Method

- **Key Technique**: Bi-Directional Feature Transformation (bFT)
- **Type**: Unpaired
- **Architecture Notes**: Introduces a bi-directional feature transformation mechanism that allows information flow between the input image and the guidance image, enabling the model to respect constraints provided by external guidance during translation.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various image domains
- **Target Modality**: Corresponding translated images guided by external inputs
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Addresses the challenge of guided image-to-image translation by effectively utilizing external guidance images.
- Demonstrates applications in pose transfer, texture transfer, and depth upsampling.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Various datasets corresponding to different translation tasks
- **Baseline Comparison**: Compared against existing conditioning mechanisms such as input concatenation and feature concatenation

### 📌 Summary Notes

- Proposes a novel conditioning mechanism that enhances the controllability and quality of image translations by leveraging bi-directional information flow between input and guidance images.
- Shows improved performance over traditional uni-directional conditioning methods.

## Unified Generative Adversarial Networks for Controllable Image-to-Image Translation (2019)

**Authors**: Hao Tang, Hong Liu, Nicu Sebe  
**Venue**: arXiv  
**Link**: [https://arxiv.org/abs/1912.06112](https://arxiv.org/abs/1912.06112)

### 🧠 Method

- **Key Technique**: Unified Generative Adversarial Network (GAN) framework
- **Type**: Unpaired
- **Architecture Notes**: Develops a single generator and discriminator model that takes a conditional image and target controllable structure as inputs, facilitating image translation guided by controllable structures such as class labels, object keypoints, human skeletons, and scene semantic maps.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various image domains
- **Target Modality**: Corresponding translated images with controllable attributes
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Enables controllable image translation by conditioning on various structural information.
- Demonstrates versatility across multiple tasks, including hand gesture translation and cross-view image translation.

### 📊 Evaluation

- **Metrics Used**: Fréchet ResNet Distance (FRD)
- **Datasets**: Datasets corresponding to hand gesture and cross-view image translation tasks
- **Baseline Comparison**: Outperforms state-of-the-art methods in controllable image translation tasks.

### 📌 Summary Notes

- Presents a unified solution for various controllable structure-guided image translation tasks.
- Emphasizes the importance of integrating structural guidance into GAN frameworks to achieve controllable and high-quality translations.

## Conditional Image-to-Image Translation (2018)

**Authors**: Jianxin Lin, Yingce Xia, Tao Qin, Zhibo Chen, Tie-Yan Liu  
**Venue**: arXiv  
**Link**: [https://arxiv.org/abs/1805.00251](https://arxiv.org/abs/1805.00251)

### 🧠 Method

- **Key Technique**: Conditional Generative Adversarial Network (cGAN) with dual learning
- **Type**: Unpaired
- **Architecture Notes**: Introduces a framework that translates an image from a source domain to a target domain conditioned on a given image in the target domain, enabling diverse and controlled translation results by leveraging domain-specific features from the conditional image.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various image domains
- **Target Modality**: Corresponding translated images influenced by conditional inputs
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Tackles the problem of generating diverse translation outputs by conditioning on target domain images.
- Utilizes dual learning to enforce consistency and improve translation quality.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Men's and women's faces, edges to shoes & bags
- **Baseline Comparison**: Compared against existing image-to-image translation methods lacking conditional diversity.

### 📌 Summary Notes

- Highlights the significance of incorporating conditional inputs from the target domain to achieve diverse and controllable image translations.
- Demonstrates the effectiveness of dual learning in enhancing translation consistency and quality.

## Conditional Image-to-Image Translation Generative Adversarial Network (cGAN) for Fabric Defect Data Augmentation (2024)

**Authors**: Swash Sami Mohammed, Hülya Gökalp Clarke  
**Venue**: Neural Computing and Applications  
**Link**: [https://link.springer.com/article/10.1007/s00521-024-10179-1](https://link.springer.com/article/10.1007/s00521-024-10179-1)

### 🧠 Method

- **Key Technique**: Conditional Generative Adversarial Network (cGAN) with U-Net architecture
- **Type**: Paired
- **Architecture Notes**: Implements a conditional U-Net generator and a 6-layered PatchGAN discriminator. The generator takes two inputs: a segmented defect mask and a clean fabric image, allowing control over various characteristics of the generated defects, such as type, shape, size, and location.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Clean fabric images
- **Target Modality**: Fabric images with synthetic defects
- **Paired/Unpaired Data**: Paired

### 🌟 Highlights

- Addresses the scarcity of comprehensive fabric defect datasets by generating realistic synthetic defect samples.
- Enhances the performance of defect detection models through data augmentation.
- The approach can be extended to other fields requiring synthetic data generation, such as medical imaging.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Fabric defect datasets
- **Baseline Comparison**: Compared against traditional data augmentation methods

### 📌 Summary Notes

- Demonstrates the effectiveness of using cGANs for controlled data augmentation in fabric defect detection.
- Highlights the potential of conditional image-to-image translation in generating diverse and realistic synthetic data for various applications.

## PAIR Diffusion: A Comprehensive Multimodal Object-Level Image Editing Framework (2024)

**Authors**: Vidit Goel, Elia Peruzzo, Yifan Jiang, Dejia Xu, Xingqian Xu, Nicu Sebe, Trevor Darrell, Zhangyang Wang, Humphrey Shi
**Venue**: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)  
**Link**: [https://cvpr.thecvf.com/virtual/2024/poster/31044](https://cvpr.thecvf.com/virtual/2024/poster/31044)

### 🧠 Method

- **Key Technique**: Diffusion-based image editing with object-level control
- **Type**: Unpaired
- **Architecture Notes**: Proposes a framework that perceives images as compositions of various objects, enabling fine-grained control over the properties of each object, specifically focusing on structure and appearance attributes.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various image domains
- **Target Modality**: Edited images with controlled object-level attributes
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Enables comprehensive editing capabilities by allowing control over individual object properties within an image.
- Utilizes diffusion models to achieve high-quality and realistic image edits.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Various datasets corresponding to different editing tasks
- **Baseline Comparison**: Compared against existing image editing methods lacking fine-grained object-level control

### 📌 Summary Notes

- Introduces a novel approach to image editing by focusing on object-level attribute control.
- Demonstrates the effectiveness of diffusion models in achieving detailed and realistic image translations.

## DCLTV: An Improved Dual-Condition Diffusion Model for Laser-Visible Image Translation (2025)

**Authors**: Xiaoyu Zhang, Laixian Zhang, Huichao Guo, Haijing Zheng, Houpeng Sun, Yingchun Li, Rong Li, Chenglong Luan, Xiaoyun Tong  
**Venue**: MDPI Sensors  
**Link**: [https://www.mdpi.com/1424-8220/25/3/697](https://www.mdpi.com/1424-8220/25/3/697)

### 🧠 Method

- **Key Technique**: Dual-Condition Diffusion Model
- **Type**: Paired
- **Architecture Notes**: Introduces an improved diffusion model that incorporates dual-condition control to guide the noise prediction process. Utilizes a Brownian bridge strategy and interpolation-based conditional injection to limit randomness and enhance translation accuracy.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Laser images
- **Target Modality**: Visible images
- **Paired/Unpaired Data**: Paired

### 🌟 Highlights

- Addresses challenges in cross-modal image translation, specifically from laser to visible spectra.
- Enhances the feature extraction capability through the integration of a self-attention mechanism.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Laser-visible image datasets
- **Baseline Comparison**: Compared against existing cross-modal image translation algorithms

### 📌 Summary Notes

- Demonstrates the effectiveness of dual-condition control in diffusion models for guided image translation.
- Highlights the potential of the proposed approach in applications requiring precise cross-modal image synthesis.

# 2.4 Text-Conditional Translation (Multimodal Relevance)

## MirrorDiffusion: Stabilizing Diffusion Process in Zero-shot Image Translation by Prompts Redescription and Beyond (2024)

**Authors**: Yupei Lin, Xiaoyu Xian, Yukai Shi, Liang Lin  
**Venue**: arXiv  
**Link**: [https://arxiv.org/abs/2401.03221](https://arxiv.org/abs/2401.03221)

### 🧠 Method

- **Key Technique**: Prompt Redescription in Diffusion Models
- **Type**: Unpaired
- **Architecture Notes**: Introduces a prompt redescription strategy to align text prompts with latent codes at each time step of the Denoising Diffusion Implicit Models (DDIM) inversion, ensuring structure-preserving reconstruction in zero-shot image translation.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Images from various domains
- **Target Modality**: Translated images guided by textual prompts
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Addresses the stochastic nature of DDPMs by stabilizing the diffusion and inversion processes.
- Enables accurate zero-shot image translation through optimized text prompts and latent code editing.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Various datasets corresponding to different translation tasks
- **Baseline Comparison**: Compared against state-of-the-art zero-shot image translation methods

### 📌 Summary Notes

- Demonstrates the effectiveness of prompt redescription in achieving stable and accurate zero-shot image translations.
- Highlights the potential of diffusion models in text-guided image translation tasks.

## FBSDiff: Plug-and-Play Frequency Band Substitution of Diffusion Features for Highly Controllable Text-Driven Image Translation (2024)

**Authors**: Xiang Gao, Jiaying Liu  
**Venue**: arXiv  
**Link**: [https://arxiv.org/abs/2408.00998](https://arxiv.org/abs/2408.00998)

### 🧠 Method

- **Key Technique**: Frequency Band Substitution in Diffusion Models
- **Type**: Unpaired
- **Architecture Notes**: Proposes a novel frequency band substitution layer that dynamically controls the reference image in text-to-image generation by decomposing guiding factors with different frequency bands of diffusion features in the DCT spectral space.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Reference images
- **Target Modality**: Images translated based on textual prompts
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Achieves flexible control over guiding factors and intensity by tuning the type and bandwidth of the substituted frequency band.
- Enables high-quality and versatile text-driven image-to-image translation without model training or fine-tuning.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Various datasets corresponding to different translation tasks
- **Baseline Comparison**: Compared against related methods in image-to-image translation

### 📌 Summary Notes

- Introduces a plug-and-play approach for integrating textual guidance into image translation tasks.
- Demonstrates the effectiveness of frequency band substitution in controlling image attributes during translation.

## One-Step Image Translation with Text-to-Image Models (2024)

**Authors**: Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, Jun-Yan Zhu
**Venue**: arXiv  
**Link**: [https://arxiv.org/abs/2403.12036](https://arxiv.org/abs/2403.12036)

### 🧠 Method

- **Key Technique**: Single-Step Diffusion Adaptation
- **Type**: Unpaired
- **Architecture Notes**: Adapts a single-step diffusion model to new tasks and domains through adversarial learning objectives, enabling image-to-image translation without paired data.

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Various image domains
- **Target Modality**: Translated images guided by text descriptions
- **Paired/Unpaired Data**: Unpaired

### 🌟 Highlights

- Addresses limitations of existing conditional diffusion models, such as slow inference speed and reliance on paired data.
- Achieves visually appealing results comparable to existing models while reducing inference steps to one.

### 📊 Evaluation

- **Metrics Used**: -
- **Datasets**: Various datasets corresponding to different translation tasks
- **Baseline Comparison**: Compared against existing conditional diffusion models

### 📌 Summary Notes

- Introduces a general method for adapting diffusion models to new tasks using adversarial learning.
- Demonstrates the potential of single-step diffusion models in efficient text-guided image translation.

# 2.5 Fast Diffusion & Distillation

## Distilling Diffusion Models into Conditional GANs (2024)

**Authors**: Minguk Kang, Richard Zhang, Connelly Barnes, Sylvain Paris, Suha Kwak, Jaesik Park, Eli Shechtman, Jun-Yan Zhu, Taesung Park  
**Venue**: ECCV
**Link**: [https://arxiv.org/abs/2405.05967](https://arxiv.org/abs/2405.05967)  
**Code**: [https://mingukkang.github.io/Diffusion2GAN/](https://mingukkang.github.io/Diffusion2GAN/)

### 🧠 Method

- **Key Technique**: Distillation of multi-step diffusion (Stable Diffusion) into one-step conditional GAN
- **Type**: Paired (generated noise-latent pairs)
- **Architecture Notes**: Uses E-LatentLPIPS (efficient latent-space perceptual loss) + multi-scale U-Net discriminator with noise/text conditioning

### 🔄 Input ↔ Output Modalities

- **Source Modality**: Gaussian noise + text prompt
- **Target Modality**: Latent (→ RGB image via decoder)
- **Paired/Unpaired Data**: Yes (noise-latent-text pairs simulated via DDIM from teacher diffusion)

### 🌟 Highlights

- Converts a slow diffusion model into a fast one-step conditional GAN (Diffusion2GAN)
- Achieves high quality image generation in 0.09s (vs. 2.59s for SD1.5, 5.60s for SDXL)
- E-LatentLPIPS enables perceptual loss without decoding to pixels, 10× faster and memory efficient
- Outperforms state-of-the-art distillation methods (SDXL-Turbo, SDXL-Lightning, InstaFlow, DMD) in FID and CLIP
- Introduces new evaluation metric: DreamSim (trajectory fidelity) and DreamDiv (diversity)

### 📊 Evaluation

- **Metrics Used**: FID, CLIP-score, Precision, Recall, DreamDiv, DreamSim
- **Datasets**: COCO2014, COCO2017, CIFAR10
- **Baseline Comparison**: SDXL-Turbo, SDXL-Lightning, InstaFlow, UFOGen, GigaGAN, Progressive Distillation

### 📌 Summary Notes

- **Strengths**: Distills a full DDIM diffusion process into a one-step generator, drastically reducing latency while maintaining quality; introduces a new latent-space perceptual loss
- **Open-source code**: ✅ [Project Page](https://mingukkang.github.io/Diffusion2GAN/)
- **Ideas to Borrow**: E-LatentLPIPS loss, noise-prompt-latent triplet generation for supervised GAN training, multi-scale latent discriminator leveraging pretrained SD weights, DreamSim for trajectory alignment evaluation

# 2.6 Cross-modal Bridge 

## An Unpaired SAR-to-Optical Image Translation Method Based on Schrödinger Bridge Network and Multi-Scale Feature Fusion (2024)  
**Authors**: Jinyu Wang, Haitao Yang, Yu He, Fengjie Zheng, Zhengjun Liu, Hang Chen  
**Venue**: Scientific Reports (Nature Publishing Group)  
**Link**: [https://doi.org/10.1038/s41598-024-75762-x](https://doi.org/10.1038/s41598-024-75762-x)

### 🧠 Method
- **Key Technique**: Schrödinger Bridge-based image translation with Multi-scale Axial Residual Module (MARM)
- **Type**: Unpaired
- **Architecture Notes**: Generator includes encoder → MARM (×9) → decoder; axial attention for local/global feature fusion; enhanced PatchGAN-style attention discriminator

### 🔄 Input ↔ Output Modalities
- **Source Modality**: SAR (Sentinel-1)
- **Target Modality**: Optical (Sentinel-2)
- **Paired/Unpaired Data**: No (Strictly unpaired training)

### 🌟 Highlights
- Novel **MARM** module extracts multi-scale, multi-orientation features using four branches and axial attention
- Schrödinger Bridge formulation enables domain interpolation without full Gaussian noise diffusion
- Significant improvement over CycleGAN, NICEGAN, UGATIT, Conditional Diffusion, and even UNSB (prior DBM)
- Ablation confirms benefit from both MARM generator and attention discriminator
- Works well even on very limited training data (as few as ~200 samples per category)

### 📊 Evaluation
- **Metrics Used**: SSIM, PSNR, LPIPS, FID
- **Datasets**:  
  - SEN1-2 (custom subset, 1659 unpaired SAR-optical image pairs from 7 land categories)
  - QXS-SAROPT (for generalization)
- **Baseline Comparison**: CycleGAN, NICEGAN, UGATIT, Conditional Diffusion, UNSB

### 📌 Summary Notes
- **Strengths**: High fidelity under limited data, strong generalization across diverse land types (e.g., desert, residence, cropland)
- **Open-source code**: ✖️ Not provided
- **Ideas to Borrow**: 
  - Use of Schrödinger Bridge loss for diffusion-based domain mapping
  - Multi-branch attention + rotation-based residual fusion (MARM)
  - SimAM + ECANet attention-enhanced discriminator for domain-aware realism checking