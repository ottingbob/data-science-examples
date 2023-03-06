import random

import gradio as gr
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# This example focuses on session state & gradio.
# In order to store data in session state we need to do 3 things:
# 1) Pass in an extra parameter into the function, which represents the state of
#   the interface
# 2) At the end of the function, return the updated value of the state as an
#   extra return value
# 3) Add the `state` input and `state` output components when creating the
#   interface


def chat(message, history):
    history = history or []
    if message.startswith("How many"):
        response = random.randint(1, 10)
    elif message.startswith("How"):
        response = random.choice(["Great", "Good", "Okay", "Bad"])
    elif message.startswith("Where"):
        response = random.choice(["Here", "There", "Somewhere"])
    else:
        response = "I don't know"
    history.append((message, str(response)))
    return history, history


"""
gr.Interface(
    fn=chat,
    # Inputs
    inputs=["text", "state"],
    # Outputs
    outputs=["chatbot", "state"],
    allow_screenshot=False,
    allow_flagging="never",
).launch()
"""

# We can use interpretation to understand model predictions. To add interpretation
# we set the `interpretation` keyword in the `Interface` class to `default`.
# This will allow users to understand what parts of the input are responsible for
# the output.
# Here is an example of an image classifier which includes interpretation:
"""
inception_net = tf.keras.applications.MobileNetV2()  # load the model

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")


def classify_image(inp):
    inp = inp.reshape((-1, 224, 224, 3))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    prediction = inception_net.predict(inp).flatten()
    return {labels[i]: float(prediction[i]) for i in range(1000)}


image = gr.Image(shape=(224, 224))
label = gr.Label(num_top_classes=3)

title = "Gradio Image Classifiction + Interpretation Example"
gr.Interface(
    fn=classify_image,
    inputs=image,
    outputs=label,
    interpretation="default",
    title=title,
).launch()
"""

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


# And another chatbot example with a model -- this does not work =(
def add_text(state, text):
    # tokenize the new input sentence
    new_user_input_ids = tokenizer.encode(
        text + tokenizer.eos_token, return_tensors="pt"
    )

    # append the new user input tokens to the chat history
    print(state)
    bot_input_ids = torch.cat([torch.LongTensor(state), new_user_input_ids], dim=-1)

    # generate a response
    history = model.generate(
        bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
    ).tolist()

    # convert the tokens to text, and then split the responses into lines
    response = tokenizer.decode(history[0]).split("<|endoftext|>")
    response = [
        (response[i], response[i + 1]) for i in range(0, len(response) - 1, 2)
    ]  # convert to tuples of list
    print(response)

    # state = state + [(text, text + "?")]
    # return state, state
    # print(response[0][1], history)
    state = state + [response[-1]]
    # return response[0][1], state
    # return state, response[0][1]
    # return state, state
    # return history, response
    # state = state + [(text, text + "?")]

    print("here is state", state)
    # return state, state
    return history, state


def add_image(state, image):
    state = state + [(f"![](/file={image.name})", "Cool pic!")]
    return state, state


with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])

    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.UploadButton("🖼️", file_types=["image"])

    # Looks like the signature is:
    # function, inputs, outputs
    txt.submit(add_text, [state, txt], [state, chatbot])
    txt.submit(lambda: "", None, txt)
    btn.upload(add_image, [state, btn], [state, chatbot])

demo.launch()
