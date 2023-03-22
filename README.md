## Data Science Scripts

This is a collection of random data-science exercises either gathered from different articles
or videos that I have used for learning.

Mileage may vary

### Usage

I setup an `arguably` complicated Makefile setup since I have so many projects in here.

Here is a demonstration of how it works for a given project:
![Demo](assets/make-demo.gif)

### Projects and Explanations

Here are some of the included projects:
- scripts/
	> one off examples usually found from blog posts demonstrating a single topic
- financial-analyst-course-2023/
	> The following directory has examples for learning financial analysis based on [this Udemy course](https://www.udemy.com/course/the-complete-financial-analyst-course)
	> The course is meant to work in excel but I have other plans...
- hugging_face_course/
	> This project follows the [hugging face course](https://huggingface.co/course/chapter0/1?fw=pt) which provides building and tweaking various types of NLP pipelines. It also exposes a little bit of [gradio](https://gradio.app/) so you can have a way to showcase your models!

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

The following examples come from the derivatives analytics with Python book
Here is the related [github repo](https://github.com/yhilpisch/dawp/blob/master/python36)
> This part deals with the market and generalizations / methods to use
- derivatives-with-python/part1
> Part 2 works with theoretical valuation
- derivatives-with-python/part2
> Part 3 dives into market based valuations
- derivatives-with-python/part3

The following examples come from the Building Machine Leaning Powered Applications Going From Idea to Product Python book
Here is the related [github repo](https://github.com/hundredblocks/ml-powered-applications)
- building-ml-powered-apps/

### Installations

During the creation of some of the projects / examples I have encountered some issues when working with my recent version of python 11. For instance tensorflow will not be happy with you and as a result it's probably a good reason to start to learn pytorch.

Anyways opinions aside (this is programming right??) here are some notes I have on some libraries having trouble with my setup.

##### umap-learn

In order to try and get `umap-learn` I needed to get a version of llvm working.

In my attempt it looks like the library is still too old to work with python 3.11 =(

```bash
$ poetry add umap-learn

FileNotFoundError: [Errno 2] No such file or directory: 'llvm-config'
RuntimeError: llvm-config failed executing, please point LLVM_CONFIG to the path for llvm-config

$ apt-cache search "llvm-.*-dev" | grep -v ocaml | sort
llvm-11-dev - Modular compiler and toolchain technologies, libraries and headers

$ sudo apt install llvm-11-dev

$ ls /usr/bin/llvm-config*
/usr/bin/llvm-config-11

$ LLVM_CONFIG=/usr/bin/llvm-config-11  poetry add umap-learn
RuntimeError: Building llvmlite requires LLVM 10.0.x or 9.0.x, got '11.1.0'. Be sure to set LLVM_CONFIG to the right executable path.
```

### Resources

Interesting [hacker news article](https://news.ycombinator.com/item?id=34971883) talks about challenges they have with AI in development scenarios. It sounds like it is mostly frustrations due to deployment / infrastructure and rapid prototyping
> We serve our models with FastAPI, containerize them, and then deploy them to our GKE clusters. Depending on the model, we choose different machines - some require GPUs, most are decent on CPU. We take models up or down based on usage, so we have cold starts unless otherwise specified by customers. We expose access to the model via a POST call through our cloud app. We track inputs and outputs, as we expect that people will become interested in fine tuning models based on their past usage.

For the original "davinci" models (now 3 generations behind if you count Instruct, ChatGPT, and upcoming DV"), OpenAI recommends "Aim for at least ~500 examples" as a starting point for fine-tuning
