### Data Science Scripts

This is a collection of random data-science exercises either gathered from different articles
or videos that I have used for learning.

Mileage may vary

#### Explainations

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
> The following file provides accelerated models for all the `main-nlp` related examples
- hugging_face_course/accelerated_models.py
