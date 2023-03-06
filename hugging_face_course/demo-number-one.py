import gradio as gr
from transformers import pipeline

# Simple Hello World Gradio Example
"""
def greet(name):
    return "Hello " + name


# Instantiate a textbox class
textbox = gr.Textbox(label="Type your name here:", placeholder="Rick James", lines=2)

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo = gr.Interface(fn=greet, inputs=textbox, outputs="text")
demo.launch()
"""

# Demo a text-generation model like GPT-2
model = pipeline("text-generation")


def predict(prompt):
    return model(prompt)[0]["generated_text"]


gr.Interface(fn=predict, inputs="text", outputs="text").launch()

# predict("My favorite programming language is")
# My favorite programming language is Java. It is more expressive and has more possibilities as a language than any other language. The best part about Java is that it makes you more likely of reading code that is easier for you to understand. You don't need
# My favorite programming language is Haskell. It is a popular language in both academia and industry. Most programmers will easily recognize it, but not everyone knows Haskell well enough to know it. So today I'm going to show you a quick look at my favorite
