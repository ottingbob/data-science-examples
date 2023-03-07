### Data Science Scripts

This is a collection of random data-science exercises either gathered from different articles
or videos that I have used for learning.

Mileage may vary

#### Explanations

The following directory has examples for learning financial analysis based on [this Udemy course](https://www.udemy.com/course/the-complete-financial-analyst-course)
The course is meant to work in excel but I have other plans...
- financial-analyst-course-2023/

- self-attention.py
> Creates an attention model by just using standard matrix multiplication / operations

The following examples are from a [pycon2020](https://www.youtube.com/watch?v=vyOgWhwUmec)
presentation with the corresponding [github](https://github.com/keithgalli/pycon2020):
> All examples are in the pycon2020-examples/ directory
> Need to run the following to get spacy dataset `poetry run spacy download en_core_web_md`
- bag-of-words.py
- word-vectors.py
- regex-ex.py
- stemming-lemmatization.py
- stopwords.py
- other-tech.py

The following example is the [hugging face course](https://huggingface.co/course/chapter0/1?fw=pt)
- hugging_face_course/transformer-models.py
- hugging_face_course/transformer-usage.py
- hugging_face_course/tuning-models.py
- hugging_face_course/datasets-lib.py
- hugging_face_course/datasets-lib-bigdata.py
- hugging_face_course/datasets-github-issues.py
- hugging_face_course/datasets-semantic-search.py
- hugging_face_course/tokenizers-quick-example.py
- hugging_face_course/tokenizers-bpe.py
- hugging_face_course/tokenizers-wp.py
- hugging_face_course/tokenizers-uni.py
- hugging_face_course/tokenizers-from-scratch.py
- hugging_face_course/main-nlp-token-classification.py
- hugging_face_course/main-nlp-masked-language.py
- hugging_face_course/main-nlp-translation.py
> In order for this one to work I needed to run the following command due to dependencies with the google/mT5 model checkpoint:
> `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python hugging_face_course/main-nlp-summarization.py`
- hugging_face_course/main-nlp-summarization.py
- hugging_face_course/main-nlp-casual-language.py
- hugging_face_course/main-nlp-question-answering.py
> The following file provides accelerated models for all the `main-nlp` related examples
- hugging_face_course/accelerated_models.py
- hugging_face_course/demo-number-one.py
- hugging_face_course/gradio-interface-class.py
- hugging_face_course/gradio-share-demos.py
- hugging_face_course/gradio-hf-hub-integration.py
- hugging_face_course/gradio-interface-advanced.py
- hugging_face_course/gradio-chat-example.py

### Resources

Interesting [hacker news article](https://news.ycombinator.com/item?id=34971883) talks about challenges they have with AI in development scenarios. It sounds like it is mostly frustrations due to deployment / infrastructure and rapid prototyping
> We serve our models with FastAPI, containerize them, and then deploy them to our GKE clusters. Depending on the model, we choose different machines - some require GPUs, most are decent on CPU. We take models up or down based on usage, so we have cold starts unless otherwise specified by customers. We expose access to the model via a POST call through our cloud app. We track inputs and outputs, as we expect that people will become interested in fine tuning models based on their past usage.

For the original "davinci" models (now 3 generations behind if you count Instruct, ChatGPT, and upcoming DV"), OpenAI recommends "Aim for at least ~500 examples" as a starting point for fine-tuning
