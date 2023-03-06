import gradio as gr

title = "GPT-J-6B"
description = "Gradio Demo for GPT-J 6B, a transformer model trained using Ben Wang's Mesh Transformer JAX. 'GPT-J' refers to the class of model, while '6B' represents the number of trainable parameters. To use it, simply add your text, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://github.com/kingoflolz/mesh-transformer-jax' target='_blank'>GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model</a></p>"
examples = [
    ["The tower is 324 metres (1,063 ft) tall,"],
    ["The Moon's orbit around Earth has"],
    ["The smooth Borealis basin in the Northern Hemisphere covers 40%"],
]

# We can use `Interface.load()` to load a model directly. Loading a model this way
# uses HF Inference API instead of loading the model in memory. This is ideal for huge
# models like GPT-J or T0pp which require lots of RAM.
"""
gr.Interface.load(
    "huggingface/EleutherAI/gpt-j-6B",
    inputs=gr.Textbox(lines=5, label="Input Text"),
    title=title,
    description=description,
    article=article,
    examples=examples,
    enable_queue=True,
).launch()
"""

# To load any Space from the HF Hub and recreate it locally we pass `spaces/` to the
# `Interface`, followed by the name of the Space.
# Here is a demo using a Space that removes the background of an image:
# gr.Interface.load("spaces/abidlabs/remove-bg").launch()

# Or use the same demo and customize it by overriding the parameters, in this case
# working with our webcam instead:
gr.Interface.load(
    "spaces/abidlabs/remove-bg", inputs="webcam", title="Remove your webcam background!"
).launch()
