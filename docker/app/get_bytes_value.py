import io
import base64


def get_bytes_value(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()
     
