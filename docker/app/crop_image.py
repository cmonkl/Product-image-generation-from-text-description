import numpy as np
from PIL import Image


def crop_image(prompt, image):
    face_included_prompts = ['shirt', 't-shirt', 'dress', 'polo', 'kuta', 'top',
                             'saree', 'sweatshirt', 'sweater', 'jacket',
                             'tunic']
    for face_prompt in face_included_prompts:
        if face_prompt in prompt.lower():
            image = Image.fromarray(np.asarray(image)[65:, :]).resize((256, 256))
            break
    return image
