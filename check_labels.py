from transformers import AutoConfig
config = AutoConfig.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
print(config.id2label)
