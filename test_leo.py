import torch
import PIL
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch
from transformers import CLIPProcessor, CLIPModel
from time import sleep


def test_clip_model(tar_files):
    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=True,
        batch_size=32
    )
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    c = 0
    while True:
        batch = get_next_batch(iter(dataset))
        if batch is None:
            break
        
        similarity_scores = []
        image_list = batch[1]
        text_list = batch[2]


        for img, txt in zip(image_list, text_list):
            inputs = processor(
                text=[txt],
                images=[img],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to("cuda")
            outputs = model(**inputs)
            score = outputs.logits_per_image.item()
            similarity_scores.append(score)


        print(f"Done with batch {c}")
        c += 1
        sleep(1)
        if c == 10:
            break