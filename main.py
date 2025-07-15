import torch
from transformers import pipeline

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    pipeline = pipeline(
        task="fill-mask",
        model="google-bert/bert-base-uncased",
        torch_dtype=torch.float16,
        device=0
    )
    pipeline("Plants create [MASK] through a process known as photosynthesis.")

