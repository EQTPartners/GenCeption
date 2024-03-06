# GenCeption: Evaluate Multimodal LLMs with Unlabeled Unimodal Data

<div>
<p align="center">
  <a href="https://github.com/EQTPartners/GenCeption/blob/main/Leaderboard.md">🔥🏅️Leaderboard🏅️🔥</a>&emsp;•&emsp;
  <a href="#contribute">Contribute</a>&emsp;•&emsp;
  <a href="https://arxiv.org/abs/2402.14973">Paper</a>&emsp;•&emsp;
  <a href="#cite-this-work">Citation</a> 
</p>

> GenCeption is an annotation-free MLLM (Multimodal Large Language Model) evaluation framework that merely requires unimodal data to assess inter-modality semantic coherence and inversely reflects the models' inclination to hallucinate.

![GenCeption Procedure](figures/genception-correlation.jpeg)

GenCeption is inspired by a popular multi-player game [DrawCeption](https://wikipedia.org/wiki/drawception). Using the image modality as an example, the process begins with a seed image $\mathbf{X}^{(0)}$ from a unimodal image dataset for the first iteration ($t$=1). The MLLM creates a detailed description of the image, which is then used by an image generator to produce $\mathbf{X}^{(t)}$. After $T$ iterations, we calculate the GC@T score to measure the MLLM's performance on $\mathbf{X}^{(0)}$. 

The GenCeption ranking on [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) benchmarking dataset (without using any label) shows a strong correlation with other sophisticated benchmarks such as [OpenCompass](https://rank.opencompass.org.cn/leaderboard-multimodal) and [HallusionBench](https://github.com/tianyi-lab/HallusionBench). Moreover, the negative correlation with MME scores suggests that GenCeption measures distinct aspects not covered by MME, using the same set of samples. For detailed experimental analysis, please read [our paper](https://arxiv.org/abs/2402.14973).

We demostrate a 5-iteration GenCeption procedure below run on a seed images to evaluate 4 VLLMs. Each iteration $t$ shows the generated image $\mathbf{X}^{(t)}$, the description $\mathbf{Q}^{(t)}$ of the preceding image $\mathbf{X}^{(t-1)}$, and the similarity score $s^{(t)}$ relative to $\mathbf{X}^{(0)}$. The GC@5 metric for each VLLM is also presented. Hallucinated elements within descriptions $\mathbf{Q}^{(1)}$ and $\mathbf{Q}^{(2)}$ as compared to the seed image are indicated with  <span style="color:red"><u>red underlined</u></span>.

![GenCeption Example](figures/existence-example.jpeg)


## Contribute
Please add your model details and results to `leaderboard/leaderboard.json` and **create a PR (Pull-Request)** to contribute your results to the [🔥🏅️**Leaderboard**🏅️🔥](https://github.com/EQTPartners/GenCeption/blob/main/leaderboard/Leaderboard.md). Start by creating your virtual environment:

```{bash}
conda create --name genception python=3.10 -y
conda activate genception
pip install -r requirements.txt
```

For example, if you want to evaluate mPLUG-Owl2 model, please follow the instructions in the [official mPLUG-OWL2 repository](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2#usage). Then run GenCeption by

```{bash}
bash example_script.sh # uses exemplary data in datasets/example/
```

This assumes that an OPENAI_API_KEY is set as an environment variable. The `model` argument to `experiment.py` in `example_script.sh` can be adjusted to `llava7b`, `llava13b`, `mPLUG`, or `gpt4v`. Please adapt accordingly for to evaluate your MLLM.

The MME dataset, of which the image modality was used in our paper, can be obtained as [described here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/Evaluation/README.md#our-mllm-works).

## Cite This Work
```bibtex
@article{cao2023genception,
    author = {Lele Cao and
              Valentin Buchner and
              Zineb Senane and
              Fangkai Yang},
    title = {{GenCeption}: Evaluate Multimodal LLMs with Unlabeled Unimodal Data},
    year={2023},
    journal={arXiv preprint arXiv:2402.14973},
    primaryClass={cs.AI,cs.CL,cs.LG}
}
```
