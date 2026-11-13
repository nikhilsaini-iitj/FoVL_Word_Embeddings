# FoVL Major Assignment: Multilingual Word Embeddings for Indian Languages

* **Student Name:** Nikhil Saini
* **Roll Number:** G25AIT2067
* **Email:** g25ait2067@iitj.ac.in
* **Course:** Foundations of Vision & Language (FoVL)
* **Academic Institution:** Indian Institute of Technology Jodhpur (IITJ)
* **Date:** November 2026
* **GitHub Repository:** [https://github.com/nikhilsaini-iitj/FoVL_Word_Embeddings](https://github.com/nikhilsaini-iitj/FoVL_Word_Embeddings)
* **Google Colab Notebook:** [G25AIT2067_FoVL_Word_Embeddings.ipynb on Google Colab](https://colab.research.google.com/drive/1JzLQm_Hel5uHo_zuY0hc4tNG-FubLj1i)
* **Training Corpus:** IndicCorpV2-5k (`rony000013/indiccorpV2-5k`)
* **Evaluation Benchmark:** Akhtar et al. (2017) Word-Similarity Datasets for Indian Languages
* **Vector Dimension:** Exactly 300 dimensions

---

## Performance Summary on Akhtar et al. (2017) Public Benchmark

| Language | Code | Scored Pairs | Model Vocab | OOV Words | Coverage | Spearman Rank (rho) | p-Value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gujarati** | `gu` | 163 | 41,355 | 23 | 89.59% | **0.4424** | 3.37e-09 |
| **Hindi** | `hi` | 235 | 27,730 | 40 | 87.42% | **0.2979** | 3.34e-06 |
| **Marathi** | `mr` | 104 | 47,662 | 68 | 79.82% | **0.2427** | 1.30e-02 |
| **Punjabi** | `pa` | 143 | 29,551 | 22 | 89.32% | **0.3892** | 1.56e-06 |
| **Tamil** | `ta` | 97 | 58,269 | 35 | 74.26% | **0.2520** | 1.28e-02 |
| **Telugu** | `te` | 111 | 49,917 | 46 | 71.25% | **0.2843** | 2.50e-03 |
| **Urdu** | `ur` | 100 | 38,006 | 26 | 82.19% | **0.3251** | 9.64e-04 |
| **Macro Average ($S_{macro}$)** | - | **953** | - | - | - | **0.3191** | **Statistically Significant** |

* **Word2Vec Baseline Macro-Average:** 0.2284
* **Joint Multilingual FastText Macro-Average:** 0.1751
* **Monolingual FastText (Proposed):** **0.3191** (+39.7% relative gain over Word2Vec baseline)

## Submission Files

* `G25AIT2067_FoVL_Word_Embeddings.ipynb`: Self-contained Colab notebook (training from scratch, OOV `get_word_vector()`, exports, and t-SNE plot)
* `G25AIT2067_FoVL_Assignment_Report.pdf`: 5-page experimental report (all 8 sections)
* `G25AIT2067_FoVL_Assignment_Report.docx`: Word version of the experimental report
* `embeddings/`: Trained word-pair `.npy` arrays of exact shape `(N, 2, 300)`
* `vocab/`: Per-language JSON vocabularies and unified `vocab_mapping.json`
* `figures/`: `tsne_all_languages.png` and `spearman_ablation_comparison.png`
* `results/evaluation_metrics.json`: Exact recomputed benchmark metrics
