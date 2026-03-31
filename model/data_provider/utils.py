import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from model.models.constants import (
    NUM_TOKENS,
    IGNORE_INDEX,
    PROPRIO_DIM
)
def prepare_input_ids(prompt_builder, conversation, print_prompt_limit, tokenizer, num_answer_tokens, predict_stop_token):
    for turn in conversation:
        prompt_builder.add_turn(turn["from"], turn["value"])

    if print_prompt_limit > 0:
        print("Conversation:", conversation)
        p = prompt_builder.get_prompt()
        print("Prompt:", p)

    # Tokenize (w/ `base_tokenizer`)
    input_ids = tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
    labels = list(input_ids)

    # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
    input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
    
    # critical, some tokenizers have different numbers of "end tokens".
    num_end_tokens = 1
    if isinstance(tokenizer, Qwen2TokenizerFast):
        # Qwen has <|im_end|><|endoftext|> for example
        num_end_tokens = 2

    # mask the input id tokens parts
    labels[0 : -(num_answer_tokens + num_end_tokens)] = IGNORE_INDEX
    if not predict_stop_token:
        labels[-num_end_tokens:] = IGNORE_INDEX

    return input_ids, labels

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
