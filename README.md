# AICS Project: Bias and Conventionality

This is the public repository for my course project for *Artificial Intelligence: Cognitive Systems*. 

Research on stereotypes propagated by multimodal LLMs (vision-language) appears relatively scarce when compared to models with only a single modality, such as vision or text (Ruggeri & Nozza, 2023). This project aims to elicit such biases through the VLStereoSet by Zhou et al. (2022), which consists of stereotypical and anti-stereotypical images, along with captions. I transform the dataset to fit an image caption matching task and intend to investigate how often a stereotypical caption is given the anti-stereotypical image. 

Datasets such as the VLStereoSet, are generated with the help of templates. Using templates for bias elicitation does not account for the richness how such content can be phrased (Dev et al., 2022). For this, I intend to investigate different methods of paraphrasing to see whether elicited phenomena are robust under perturbations.


## Literature Review

[On Measures of Biases and Harms in NLP](https://aclanthology.org/2022.findings-aacl.24) (Dev et al., Findings 2022)

[A Multi-dimensional study on Bias in Vision-Language models](https://aclanthology.org/2023.findings-acl.403) (Ruggeri & Nozza, Findings 2023)

[VLStereoSet: A Study of Stereotypical Bias in Pre-trained Vision-Language Models](https://aclanthology.org/2022.aacl-main.40) (Zhou et al., AACL-IJCNLP 2022)