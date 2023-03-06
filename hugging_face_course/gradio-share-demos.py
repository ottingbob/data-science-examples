from pathlib import Path

import gradio as gr
import torch
from torch import nn
from transformers import pipeline

# Mainly doing this example to learn more about the gradio components.
# This will not end up being permanently hosted on `Spaces`
generator = pipeline("text-generation", model="huggingtweets/rickandmorty")


# We can use the following parameters with the `Interface`:
# `title`: gives a title to the demo which appears above input / output components
# `description`: description for the interface appearing above i/o components
# `article`: explains the interface, appears below i/o components
# `theme`: set the theme of the colours / styling used. Can use the `dark-` prefix
# `examples`: example inputs to the function & populate the interface
#   provided as a nested list of examples to input per component
# `live`: model reruns every time the input changes
def predict(prompt):
    return generator(prompt, num_return_sequences=5)


title = "Ask Rick a Question"
description = """
The bot was trained to answer questions based on Rick and Morty dialogues. Ask Rick anything!
<img src="https://huggingface.co/spaces/course-demos/Rick_and_Morty_QA/resolve/main/rick.png" width=200px>
"""

article = "Check out [the original Rick and Morty Bot](https://huggingface.co/spaces/kingabzpro/Rick_and_Morty_Bot) that this demo is based off of."

"""
gr.Interface(
    fn=predict,
    inputs="textbox",
    outputs="text",
    title=title,
    description=description,
    article=article,
    examples=[["What are you doing?"], ["Where should we time travel to?"]],
).launch()
"""
# Can optionally share a link to our interface by using:
# ).launch(share=True)

# And now create an example with sketch recognition
# We load the labels from `class_names.txt` and load the pre-trained pytorch model
LABELS = Path("hugging_face_course/class_names.txt").read_text().splitlines()

model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1152, 256),
    nn.ReLU(),
    nn.Linear(256, len(LABELS)),
)
state_dict = torch.load(
    "hugging_face_course/sketch_recognition_pytorch_model.bin", map_location="cpu"
)
model.load_state_dict(state_dict, strict=False)
model.eval()


def predict_sketch(im):
    x = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        out = model(x)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    values, indices = torch.topk(probabilities, 5)
    return {LABELS[i]: v.item() for i, v in zip(indices, values)}


gr.Interface(
    predict_sketch,
    inputs="sketchpad",
    outputs="label",
    # Currently this is deprecated ??
    theme="huggingface",
    title="Sketch Recognition",
    description="Who wants to play Pictionary? Draw a common object like a shovel or a laptop, and the algorithm will guess in real time!",
    article="<p style='text-align: center'>Sketch Recognition | Demo Model</p>",
    live=True,
).launch()
