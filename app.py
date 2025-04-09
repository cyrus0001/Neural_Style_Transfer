import os
import gradio as gr
from fast_neural_style_transfer import fast_neural_style_transfer
from classical_neural_style_transfer import classical_neural_style_transfer
import tensorflow as tf
from PIL import Image
import tempfile

def style_transfer_interface(content_imgs, style_img, model_type, custom_model_file=None, use_gpu=False):
    # Restrict GPU if requested
    if not use_gpu:
        try:
            tf.config.set_visible_devices([], 'GPU')
        except:
            pass

    results = []
    model_path = os.path.abspath(custom_model_file.name) if custom_model_file else None

    for content_img_path in content_imgs:
        content_img = Image.open(content_img_path)
        style_img_pil = Image.fromarray(style_img)

        if model_type == "Classic Neural Style Transfer":
            # Save images temporarily for classical model which uses paths
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as c_img, \
                 tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as s_img:
                content_img.save(c_img.name)
                style_img_pil.save(s_img.name)
                result_array = classical_neural_style_transfer(c_img.name, s_img.name, iterations=1000)
                result = Image.fromarray(result_array)
        else:
            result = fast_neural_style_transfer(content_img, style_img_pil, model_path)

        results.append(result)

    return results if len(results) > 1 else results[0]

# Gradio UI
interface = gr.Interface(
    fn=style_transfer_interface,
    inputs=[
        gr.File(label="Upload Content Images", type="filepath", file_types=[".png", ".jpg", ".jpeg"], file_count="multiple"),
        gr.Image(label="Style Image", type="numpy"),
        gr.Radio(["Classic Neural Style Transfer", "Fast Neural Style Transfer"], label="Select Model"),
        gr.File(label="Upload Custom TF Hub Model (.tar.gz or directory)", file_types=[".tar.gz", None]),
        gr.Checkbox(label="Use GPU", value=True)
    ],
    outputs=gr.Gallery(label="Stylized Output"),
    title="🎨 Neural Style Transfer",
    description="Upload one or more content images and a style image. Choose a model type, optionally upload a custom TF Hub model, and toggle GPU usage."
)


if __name__ == '__main__':
    interface.launch(share=True)

