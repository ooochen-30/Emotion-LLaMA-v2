# MMEVerse-Bench

**MMEVerse-Bench** is a large-scale, unified benchmark for **multimodal affective intelligence**, designed to systematically evaluate models across diverse affect-related tasks and real-world settings.

Compared with the prior **MERR** dataset, **MMEVerse** expands the data scale by **more than 3×** and reorganizes all re-annotated samples into a coherent benchmark suite, **MMEVerse-Bench**, covering **18 evaluation sets** across four major affective understanding tasks:

* **Emotion Recognition**
* **Sentiment Analysis**
* **Intent Prediction**
* **Multimodal Reasoning**

This benchmark provides a comprehensive and challenging evaluation platform for studying **robust, generalizable, and trustworthy multimodal affective models**.

All samples are strictly organized into **MMEVerse-Train** and **MMEVerse-Bench**, while **preserving the original training, validation, and test splits** of each source dataset. This design ensures:

* strict data separation,
* reproducibility across studies,
* and effective prevention of information leakage.

Importantly, because **emotional taxonomies and annotation schemes vary substantially across datasets**, MMEVerse-Bench adopts a **dataset-native evaluation protocol**. Each evaluation set retains its original label space instead of forcing category alignment, enabling:

* faithful representation of real-world affective diversity,
* fair cross-dataset comparison,
* and realistic assessment of model generalization.

---

## Benchmark Composition

MMEVerse-Bench integrates representative datasets across multiple affective understanding paradigms:

### 🔹 Basic Emotion

* MER2023
* MER2024
* MELD-e
* IEMOCAP
* CAER
* E3
* DFEW
* MAFW
* MC-EIU

### 🔹 Sentiment

* CMU-MOSI
* CMU-MOSEI
* CH-SIMS
* CH-SIMS-v2
* MELD-s

### 🔹 Multi-Label Emotion

* MAFW-m
* BOLD

### 🔹 Open-Vocabulary Emotion

* OV-MERD+

### 🔹 Intention Prediction

* MC-EIU-i

---

## Basic Emotion

**Basic emotion recognition** focuses on identifying discrete emotional categories (e.g., *happy, sad, angry, neutral*) from multimodal signals such as facial expressions, vocal tone, body movements, and spoken language.
This task forms the foundation of affective computing and serves as a critical testbed for evaluating multimodal fusion, temporal modeling, and robustness to noise and modality imbalance.

MMEVerse-Bench includes the following widely-used basic emotion datasets:

* **Dataset**: MER2023
  **Paper**: *MER 2023: Multi-label Learning, Modality Robustness, and Semi-Supervised Learning*
  **Link**: [https://huggingface.co/datasets/MERChallenge/MER2023](https://huggingface.co/datasets/MERChallenge/MER2023)

* **Dataset**: MER2024
  **Paper**: *MER 2024: Semi-Supervised Learning, Noise Robustness, and Open-Vocabulary Multimodal Emotion Recognition*
  **Link**: [https://huggingface.co/datasets/MERChallenge/MER2024](https://huggingface.co/datasets/MERChallenge/MER2024)

* **Dataset**: MELD-e
  **Paper**: *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation*
  **Link**: [https://github.com/declare-lab/MELD](https://github.com/declare-lab/MELD)

* **Dataset**: IEMOCAP
  **Paper**: *IEMOCAP: Interactive Emotional Dyadic Motion Capture Database*
  **Link**: [https://sail.usc.edu/iemocap/](https://sail.usc.edu/iemocap/)

* **Dataset**: CAER
  **Paper**: *Context-Aware Emotion Recognition Networks*
  **Link**: [https://caer-dataset.github.io/](https://caer-dataset.github.io/)

* **Dataset**: E3
  **Paper**: *E³: Exploring Embodied Emotion Through a Large-Scale Egocentric Video Dataset*
  **Link**: [https://github.com/Exploring-Embodied-Emotion-official/E3](https://github.com/Exploring-Embodied-Emotion-official/E3)

* **Dataset**: DFEW
  **Paper**: *DFEW: A Large-Scale Database for Recognizing Dynamic Facial Expressions in the Wild*
  **Link**: [https://dfew-dataset.github.io/download.html](https://dfew-dataset.github.io/download.html)

* **Dataset**: MAFW
  **Paper**: *MAFW: A Large-scale, Multi-modal, Compound Affective Database for Dynamic Facial Expression Recognition in the Wild*
  **Link**: [https://mafw-database.github.io/MAFW/](https://mafw-database.github.io/MAFW/)

* **Dataset**: MC-EIU
  **Paper**: *Emotion and Intent Joint Understanding in Multimodal Conversation: A Benchmarking Dataset*
  **Link**: [https://huggingface.co/datasets/YulangZhuo/MC-EIU](https://huggingface.co/datasets/YulangZhuo/MC-EIU)

---

## Sentiment

**Multimodal sentiment analysis** aims to infer users’ **attitudes and opinions** (e.g., positive, negative, neutral) by jointly modeling **visual behaviors, vocal cues, and linguistic content**.
Compared to emotion recognition, sentiment analysis emphasizes **subjective polarity, opinion intensity, and contextual semantics**, making it especially relevant for opinion mining, human–computer interaction, and social media analysis.

By integrating complementary cues across modalities, sentiment models are expected to achieve more accurate and robust understanding of user intent and affective stance in real-world scenarios.

MMEVerse-Bench includes the following multimodal sentiment datasets:

* **Dataset**: CMU-MOSEI
  **Paper**: *Multimodal Language Analysis in the Wild*
  **Link**: [http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)

* **Dataset**: CMU-MOSI
  **Paper**: *MOSI: Multimodal Corpus of Sentiment Intensity and Subjectivity Analysis in Online Opinion Videos*
  **Link**: [http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)

* **Dataset**: CH-SIMS
  **Paper**: *CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotation of Modality*
  **Link**: [https://thuiar.github.io/sims.github.io/chsims](https://thuiar.github.io/sims.github.io/chsims)

* **Dataset**: CH-SIMS v2.0
  **Paper**: *Make Acoustic and Visual Cues Matter: CH-SIMS v2.0 Dataset and AV-Mixup Consistent Module*
  **Link**: [https://thuiar.github.io/sims.github.io/chsims](https://thuiar.github.io/sims.github.io/chsims)

* **Dataset**: MELD-s
  **Paper**: *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation*
  **Link**: [https://github.com/declare-lab/MELD](https://github.com/declare-lab/MELD)

---

## Multi-Label Emotion

In realistic affective scenarios, human emotions are often **complex, overlapping, and co-occurring**, making single-label annotation insufficient.
**Multi-label emotion recognition** allows models to predict multiple emotional states simultaneously, providing a more nuanced and faithful representation of human affect.

This task challenges models to reason about emotional composition, intensity overlap, and subtle affective mixtures across modalities.

MMEVerse-Bench includes:

* **Dataset**: BOLD
  **Paper**: *ARBEE: Towards Automated Recognition of Bodily Expression of Emotion in the Wild*
  **Link**: [https://cydar.ist.psu.edu/emotionchallenge/dataset.php](https://cydar.ist.psu.edu/emotionchallenge/dataset.php)

* **Dataset**: MAFW-m
  **Paper**: *MAFW: A Large-scale, Multi-modal, Compound Affective Database for Dynamic Facial Expression Recognition in the Wild*
  **Link**: [https://mafw-database.github.io/MAFW/](https://mafw-database.github.io/MAFW/)

---

## OV-Emotion (Open-Vocabulary Emotion)

**Open-vocabulary emotion recognition** moves beyond fixed emotion categories and requires models to recognize **previously unseen or free-form emotion concepts** described in natural language.
This setting better reflects real-world affective diversity and tests the **semantic generalization and language grounding abilities** of multimodal models.

* **Dataset**: OV-MERD+
  **Paper**: *OV-MER: Towards Open-Vocabulary Multimodal Emotion Recognition*
  **Link**: [https://huggingface.co/datasets/MERChallenge/MER2025](https://huggingface.co/datasets/MERChallenge/MER2025)

---

## Intention

**Intent prediction** focuses on understanding **speaker goals, communicative intent, and interaction purpose** in multimodal conversations.
By jointly modeling emotional cues and contextual semantics, this task bridges affective understanding and decision-oriented reasoning.

* **Dataset**: MC-EIU
  **Paper**: *Emotion and Intent Joint Understanding in Multimodal Conversation: A Benchmarking Dataset*
  **Link**: [https://huggingface.co/datasets/YulangZhuo/MC-EIU](https://huggingface.co/datasets/YulangZhuo/MC-EIU)

---

## 📖 Citation

If you use the **MME-Emotion benchmark** or find any of the following datasets helpful for your research, please consider citing the corresponding papers:

```bibtex
@article{busso2008iemocap,
  title={IEMOCAP: Interactive emotional dyadic motion capture database},
  author={Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and Chang, Jeannette N and Lee, Sungbok and Narayanan, Shrikanth S},
  journal={Language Resources and Evaluation},
  volume={42},
  pages={335--359},
  year={2008},
  publisher={Springer}
}

@inproceedings{lee2019context,
  title={Context-Aware Emotion Recognition Networks},
  author={Lee, Jiyoung and Kim, Seungryong and Kim, Sunok and Park, Jungin and Sohn, Kwanghoon},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10143--10152},
  year={2019}
}

@article{liu2024emotion,
  title={Emotion and Intent Joint Understanding in Multimodal Conversation: A Benchmarking Dataset},
  author={Liu, Rui and Zuo, Haolin and Lian, Zheng and Xing, Xiaofen and Schuller, Bj{\"o}rn W and Li, Haizhou},
  journal={arXiv preprint arXiv:2407.02751},
  year={2024}
}

@article{feng20243,
  title={$E^3$: Exploring Embodied Emotion Through A Large-Scale Egocentric Video Dataset},
  author={Feng, Yueying and Han, WenKang and Jin, Tao and Zhao, Zhou and Wu, Fei and Yao, Chang and Chen, Jingyuan and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={118182--118197},
  year={2024}
}

@article{poria2018meld,
  title={MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations},
  author={Poria, Soujanya and Hazarika, Devamanyu and Majumder, Navonil and Naik, Gautam and Cambria, Erik and Mihalcea, Rada},
  journal={arXiv preprint arXiv:1810.02508},
  year={2018}
}

@inproceedings{jiang2020dfew,
  title={DFEW: A Large-Scale Database for Recognizing Dynamic Facial Expressions in the Wild},
  author={Jiang, Xingxun and Zong, Yuan and Zheng, Wenming and Tang, Chuangao and Xia, Wanchuang and Lu, Cheng and Liu, Jiateng},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2881--2889},
  year={2020}
}

@inproceedings{liu2022mafw,
  title={MAFW: A Large-Scale, Multi-Modal, Compound Affective Database for Dynamic Facial Expression Recognition in the Wild},
  author={Liu, Yuanyuan and Dai, Wei and Feng, Chuanxu and Wang, Wenbin and Yin, Guanghao and Zeng, Jiabei and Shan, Shiguang},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={24--32},
  year={2022}
}

@inproceedings{lian2024mer,
  title={MER 2024: Semi-Supervised Learning, Noise Robustness, and Open-Vocabulary Multimodal Emotion Recognition},
  author={Lian, Zheng and Sun, Haiyang and Sun, Licai and Wen, Zhuofan and Zhang, Siyuan and Chen, Shun and Gu, Hao and Zhao, Jinming and Ma, Ziyang and Chen, Xie and others},
  booktitle={Proceedings of the 2nd International Workshop on Multimodal and Responsible Affective Computing},
  pages={41--48},
  year={2024}
}

@article{luo2020arbee,
  title={ARBEE: Towards Automated Recognition of Bodily Expression of Emotion in the Wild},
  author={Luo, Yu and Ye, Jianbo and Adams, Reginald B and Li, Jia and Newman, Michelle G and Wang, James Z},
  journal={International Journal of Computer Vision},
  volume={128},
  pages={1--25},
  year={2020},
  publisher={Springer}
}

@inproceedings{zadeh2018multimodal,
  title={Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph},
  author={Zadeh, AmirAli Bagher and Liang, Paul Pu and Poria, Soujanya and Cambria, Erik and Morency, Louis-Philippe},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={2236--2246},
  year={2018}
}

@article{zadeh2016mosi,
  title={MOSI: Multimodal Corpus of Sentiment Intensity and Subjectivity Analysis in Online Opinion Videos},
  author={Zadeh, Amir and Zellers, Rowan and Pincus, Eli and Morency, Louis-Philippe},
  journal={arXiv preprint arXiv:1606.06259},
  year={2016}
}

@inproceedings{liu2022make,
  title={Make Acoustic and Visual Cues Matter: CH-SIMS v2.0 Dataset and AV-Mixup Consistent Module},
  author={Liu, Yihe and Yuan, Ziqi and Mao, Huisheng and Liang, Zhiyun and Yang, Wanqiuyue and Qiu, Yuanzhe and Cheng, Tie and Li, Xiaoteng and Xu, Hua and Gao, Kai},
  booktitle={Proceedings of the 2022 International Conference on Multimodal Interaction},
  pages={247--258},
  year={2022}
}

@inproceedings{yu2020ch,
  title={Ch-sims: A chinese multimodal sentiment analysis dataset with fine-grained annotation of modality},
  author={Yu, Wenmeng and Xu, Hua and Meng, Fanyang and Zhu, Yilin and Ma, Yixiao and Wu, Jiele and Zou, Jiyun and Yang, Kaicheng},
  booktitle={Proceedings of the 58th annual meeting of the association for computational linguistics},
  pages={3718--3727},
  year={2020}
}

@article{lian2024ov,
  title={OV-MER: Towards Open-Vocabulary Multimodal Emotion Recognition},
  author={Lian, Zheng and Sun, Haiyang and Sun, Licai and Chen, Haoyu and Chen, Lan and Gu, Hao and Wen, Zhuofan and Chen, Shun and Zhang, Siyuan and Yao, Hailiang and others},
  journal={arXiv preprint arXiv:2410.01495},
  year={2024}
}

@inproceedings{lian2023mer,
  title={Mer 2023: Multi-label learning, modality robustness, and semi-supervised learning},
  author={Lian, Zheng and Sun, Haiyang and Sun, Licai and Chen, Kang and Xu, Mingyu and Wang, Kexin and Xu, Ke and He, Yu and Li, Ying and Zhao, Jinming and others},
  booktitle={Proceedings of the 31st ACM international conference on multimedia},
  pages={9610--9614},
  year={2023}
}
```

