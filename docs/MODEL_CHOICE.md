# Model Choice

## Primary Recommendation

Use `MahmoodLab/UNI` from Hugging Face as a frozen tile encoder, then train a
small attention MIL Cox survival head on cached tile embeddings.

This is the best fit for the first version because the endpoint task is TCGA
survival prediction. Models pretrained directly on TCGA tiles are useful for
prototyping, but they create a leakage concern when the downstream benchmark is
also TCGA. UNI is gated, but its model card states that public datasets such as
TCGA, CPTAC, PANDA, and TCIA were excluded from pretraining.

## Hardware Plan

For a 16 GB MacBook Air:

- Freeze the encoder.
- Extract features once and save `.npy` files.
- Use small encoder batches, usually 1 to 4.
- Train the MIL survival head from cached features.
- Limit bags with `--max-tiles` during development.

Do not attempt end-to-end WSI fine-tuning on this laptop.

## Practical Fallback

If UNI access is not available, start with `owkin/phikon` because it is much
smaller and runs comfortably on laptop hardware. Treat it as a prototype encoder
for TCGA, not as a clean held-out TCGA benchmark.

Useful model-card links:

- `MahmoodLab/UNI`: https://huggingface.co/MahmoodLab/UNI
- `owkin/phikon`: https://huggingface.co/owkin/phikon
- `owkin/phikon-v2`: https://huggingface.co/owkin/phikon-v2
- GDC API field reference: https://docs.gdc.cancer.gov/API/Users_Guide/Appendix_A_Available_Fields/

## Survival Model

The model in this repository is:

```text
tile image -> frozen HF pathology encoder -> tile embeddings
tile embeddings -> gated attention MIL -> slide embedding
slide embedding + optional clinical covariates -> Cox risk score
```

The loss is Cox partial likelihood. Validation uses a censored concordance index.
