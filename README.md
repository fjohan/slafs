# Swedish Lemma Animacy and Frequency Sampling

This project extracts **Swedish nouns**, links them to **SALDO** lexical entries to determine **animacy** (animate vs. inanimate), and then draws **frequency-stratified samples** for linguistic or psycholinguistic analysis.

---

## 🧩 Overview

**Goal:**  
Obtain frequency-balanced sets of animate and inanimate Swedish nouns with their SALDO semantic paths.

**Main data sources:**
- `stats_all.txt.zip` — large tab-separated corpus frequency file (from Språkbanken)
- `saldo.xml` — Swedish lexicon in LMF format

---

## ⚙️ Pipeline

### 1. Extract noun frequencies

The raw corpus file is huge.  
To avoid decompressing it entirely, stream it through `zcat`, keep only **nouns (NN)**, and discard very rare words.

```bash
zcat stats_all.txt.zip   | awk -F'\t' '$2 ~ /^NN/ && length($3) > 1 {print}'   | head -1000000   | awk -F'\t' '$5 > 100 {print}'   > stats_min100.txt
```

- Keeps the first 1,000,000 noun rows (enough to exclude most hapax legomena).  
- Keeps only items with **frequency > 100** (column 5).  
- Writes the filtered sample to `stats_min100.txt`.

---

### 2. Link with SALDO and compute animacy

Use the `build_lemma_freq_animacy.py` script to:
- parse `saldo.xml`,
- trace each noun’s **semantic parent chain** (`primary` relations),
- mark lemmas as **animate** if they ultimately descend from `människa`, `person`, `djur`, or `varelse`,
- and output a table of frequencies with animacy labels and full semantic paths.

```bash
python3 build_lemma_freq_animacy.py   --saldo-xml saldo.xml   --stats stats_min100.txt   --out lemma_freq_animacy_paths.tsv
```

**Output:**  
`lemma_freq_animacy_paths.tsv` with columns:

| writtenForm | lemgram | frequency | animacy | path |
|--------------|----------|------------|----------|------|
| djur | djur..nn.1 | 123456 | animate | djur → varelse |
| ansvar | ansvar..nn.1 | 1455028 | inanimate | ansvar → … |

---

### 3. Draw frequency-stratified samples

To get comparable animate / inanimate sets (balanced across frequency ranges):

```bash
python3 sample_stratified_animacy.py   --tsv lemma_freq_animacy_paths.tsv   --n 50   --seed 43
```

This produces:
- `sampled_animate.tsv` – 50 animate nouns  
- `sampled_inanimate.tsv` – 50 inanimate nouns  

Each file includes a mix of high-, mid-, and low-frequency words, so the overall frequency distributions of the two groups are similar.

---

## 🧾 Files produced

| File | Description |
|------|--------------|
| `stats_min100.txt` | Filtered noun frequencies from the corpus |
| `lemma_freq_animacy_paths.tsv` | Lemmas with summed frequency, animacy, and semantic path |
| `sampled_animate.tsv` | Frequency-stratified animate sample |
| `sampled_inanimate.tsv` | Frequency-stratified inanimate sample |
| `unmatched_lemgrams.txt` | Lemgrams not found in SALDO (diagnostics) |

---

## 🧠 Notes

- `normalize_key()` in `build_lemma_freq_animacy.py` automatically cleans pipe-wrapped compound lemgrams such as `|förslag_2 |förslag_2..nn.1|förslag..nn.1|`, extracting the canonical `förslag..nn.1` so SALDO lookups work correctly.
- Both scripts accept compressed input (`.gz` or single-file `.zip`).
- Adjust the frequency threshold or sample size to suit your corpus scale.

---

**Author:**  
Your Name – 2025  
Licensed under MIT / CC-BY as appropriate.
