# (PART\*) Share and Deploy {-}

# Upload to Huggingface
# Deploy Model with torchserve


```bash
torch-model-archiver \
    --model-name text_model \
    --version 1.0 \
    --serialized-file motions.pt \
    --handler text_classifier

mkdir model_store
mv text_model.mar model_store/
```
