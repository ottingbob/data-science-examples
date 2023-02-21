from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(
    classifier(
        [
            "I've been waiting for a HuggingFace course my whole life.",
            "I hate this so much!",
        ]
    )
)

# Three main steps happen when you pass text to a pipeline:
# 1) Text is preprocessed into a format the model can understand
# 2) Preprocessed inputs are passed to the model
# 3) Predictions of the model are post-processed, so you can make sense of them


# Zero-shot classification
# Here we classify texts that haven't been labelled.
# This allows us to specify which labels to use for classification
classifier = pipeline("zero-shot-classification")
res = classifier(
    [
        "This is a course about the Transformers library",
        "This should be classified as cruel and unusual punishment",
    ],
    candidate_labels=["education", "politics", "business"],
)
print(res)

# Text Generation
# You provide a prompt and the model will auto-complete it by generating
# the remaining text. Similar to predictive text & results may not be consistent
# across subsequent runs
generator = pipeline("text-generation")
res = generator("In this course, we will teach you how to")
print(res)
# [{'generated_text': 'In this course, we will teach you how to write your own code, and develop your own framework to simplify building your codebase. We will be working to build a web framework over some of the best open source frameworks around, and then will cover'}]

# Here we do text generation using a specific model, `distilgpt2`
generator = pipeline("text-generation", model="distilgpt2")
res = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print(res)
# [{'generated_text': 'In this course, we will teach you how to convert your knowledge into money – in the form of real time marketing, so we will help you with'}, {'generated_text': 'In this course, we will teach you how to take a closer look at the main features:\n\n\n\n\n\n\n\n\n\n\n'}]

# Mask filling
# The fill-mask pipeline fills in the blanks in a given text
# The `top_k` arg controls how many possibilities displayed for the masked token
unmasker = pipeline("fill-mask")
res = unmasker("This couse will teach you all about <mask> models", top_k=2)
print(res)
# [{'score': 0.09319697320461273, 'token': 774, 'token_str': ' role', 'sequence': 'This couse will teach you all about role models'}, {'score': 0.05577169731259346, 'token': 3477, 'token_str': ' animal', 'sequence': 'This couse will teach you all about animal models'}]

# Named entity recognition
# NER is a task where the model finds which parts of the input text corresponds
# to entities such as persons, locations, or organizations.
ner = pipeline("ner", grouped_entities=True)
res = ner("My name is Robert and I work in Secops in Michigan.")
print(res)
# [
#   {'entity_group': 'PER', 'score': 0.9990528, 'word': 'Robert', 'start': 11, 'end': 17},
#   {'entity_group': 'ORG', 'score': 0.98315114, 'word': 'Secops', 'start': 32, 'end': 38},
#   {'entity_group': 'LOC', 'score': 0.999087, 'word': 'Michigan', 'start': 42, 'end': 50}
# ]

# The `grouped_entities` arg tells the pipeline to regroup together the parts of
# the sentence that correspond to the same entity.

# Question Answering
# Answers questions using information from a given context. It does not generate the
# answer it extracts information from the provided context.
qa = pipeline("question-answering")
res = qa(
    question="Where do I work?",
    context="My name is Robert and I work in Secops in Michigan.",
)
print(res)
# {'score': 0.5777071714401245, 'start': 32, 'end': 38, 'answer': 'Secops'}

# Summarization
# This pipeline reduces text while keeping all (or most) of the important aspects
# referenced in the text.
summarizer = pipeline("summarization")
res = summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of
    graduates in traditional engineering disciplines such as mechanical, civil,
    electrical, chemical, and aeronautical engineering declined, but in most of
    the premier American universities engineering curricula now concentrate on
    and encourage largely the study of engineering science. As a result, there
    are declining offerings in engineering subjects dealing with infrastructure,
    the environment, and related issues, and greater concentration on high
    technology subjects, largely supporting increasingly complex scientific
    developments. While the latter is important, it should not be at the expense
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other
    industrial countries in Europe and Asia, continue to encourage and advance
    the teaching of engineering. Both China and India, respectively, graduate
    six and eight times as many traditional engineers as does the United States.
    Other industrial countries at minimum maintain their output, while America
    suffers an increasingly serious decline in the number of engineering graduates
    and a lack of well-educated engineers.
    """
)
print(res)
# [{'summary_text': ' China and India graduate six and eight times as many traditional engineers as the U.S. as does other industrial countries . America suffers an increasingly serious decline in the number of engineering graduates and a lack of well-educated engineers . There are declining offerings in engineering subjects dealing with infrastructure, infrastructure, the environment, and related issues .'}]

# Translation
# Easiest way to translate is to use a model for the language conversion that you are
# targeting. In this case this is one optimized for french to english
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
res = translator("Ce cours est produit par Hugging Face.")
print(res)
# [{'translation_text': 'This course is produced by Hugging Face.'}]

# There can also be bias in the pretrained data sets. In this example we see
# the model only give 1 gender free example and `prostitute` ended up in the
# top 5 possibilities for women and work...
unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])
# ['carpenter', 'lawyer', 'farmer', 'businessman', 'doctor']

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])
# ['nurse', 'maid', 'teacher', 'waitress', 'prostitute']
