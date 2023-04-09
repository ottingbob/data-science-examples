## Data Science Scripts

This is a collection of random data-science exercises either gathered from different articles
or videos that I have used for learning.

<div align="center">
	<strong>
		<i>___Mileage may vary___</i>
	</strong>
	<br />
	<img src="assets/ggf.gif" width="200" />
</div>

### Usage

I setup an ___arguably___ complicated Makefile structure since I have so many projects in here.

Here is a demonstration of how it works for the related `building_ml_powered_apps` project:
![Demo](assets/make-demo.gif)

### Projects and Explanations

Here are some of the included projects:
- [building_ml_powered_apps/](https://github.com/ottingbob/data-science-examples/tree/main/building_ml_powered_apps)
	> This project is built from the Building Machine Leaning Powered Applications Going From Idea to Product Python book. Here is the related [github repo](https://github.com/hundredblocks/ml-powered-applications)
- [derivatives-with-python/](https://github.com/ottingbob/data-science-examples/tree/main/derivatives-with-python)
	> The following examples come from the Derivatives Analytics With Python book. Here is the related [github repo](https://github.com/yhilpisch/dawp/blob/master/python36)
- [financial_analyst_course_2023/](https://github.com/ottingbob/data-science-examples/tree/main/financial_analyst_course_2023)
	> The following directory has examples for learning financial analysis based on [this Udemy course](https://www.udemy.com/course/the-complete-financial-analyst-course)
	>
	> The course is meant to work in excel but I have other plans...
- [hugging_face_course/](https://github.com/ottingbob/data-science-examples/tree/main/hugging_face_course)
	> This project follows the [hugging face course](https://huggingface.co/course/chapter0/1?fw=pt) which provides building and tweaking various types of NLP pipelines. It also exposes a little bit of [gradio](https://gradio.app/) so you can have a way to showcase your models!
- [pycon2020-examples/](https://github.com/ottingbob/data-science-examples/tree/main/pycon2020-examples)
	> This collection of scripts is based off [Keith Galli's](https://github.com/keithgalli) awesome [PyCon2020 NLP youtube video](https://www.youtube.com/watch?v=vyOgWhwUmec). It runs through various NLP techniques and serves as a good introduction with some guidance for starting off your ML journey
- [python_data_science_december_2022/](https://github.com/ottingbob/data-science-examples/tree/main/python_data_science_december_2022)
	> These are a collection of activities that do some basic data analysis on a wide variety of sources
- [scripts/](https://github.com/ottingbob/data-science-examples/tree/main/scripts)
	> one off examples usually found from blog posts demonstrating a single topic

### Installations

##### gum

[gum](https://github.com/charmbracelet/gum) is a library to make good-looking & interactive shell scripts. I use it on some of the scripts in here.

I have included the installation for my system below:
```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.charm.sh/apt/gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/charm.gpg
echo "deb [signed-by=/etc/apt/keyrings/charm.gpg] https://repo.charm.sh/apt/ * *" | sudo tee /etc/apt/sources.list.d/charm.list
sudo apt update && sudo apt install gum
```

##### problematic installations

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

An interesting thing to check out could be [this python profiler py-spy](https://github.com/benfred/py-spy) which allows you to profile and debug any running python program, even if it is serving production traffic

[Face recognition API](https://github.com/ageitgey/face_recognition) in python

Deep learning [lessons](https://course.fast.ai/) from Fast AI

Interesting transaction manager called [pyWave](https://github.com/therealOri/pyWave)

Tool called [Oxen](https://github.com/oxen-ai/oxen-release) for versioned datasets

Interesting [hacker news article](https://news.ycombinator.com/item?id=34971883) talks about challenges they have with AI in development scenarios. It sounds like it is mostly frustrations due to deployment / infrastructure and rapid prototyping
> We serve our models with FastAPI, containerize them, and then deploy them to our GKE clusters. Depending on the model, we choose different machines - some require GPUs, most are decent on CPU. We take models up or down based on usage, so we have cold starts unless otherwise specified by customers. We expose access to the model via a POST call through our cloud app. We track inputs and outputs, as we expect that people will become interested in fine tuning models based on their past usage.

For the original "davinci" models (now 3 generations behind if you count Instruct, ChatGPT, and upcoming DV"), OpenAI recommends "Aim for at least ~500 examples" as a starting point for fine-tuning

How do you monitor and debug models?
> When your engineering team builds great tooling, monitoring and
debugging get much easier. Stitch Fix has built an internal tool that takes
in a modeling pipeline and creates a Docker container, validates arguments
and return types, exposes the inference pipeline as an API, deploys it on
our infrastructure, and builds a dashboard on top of it. This tooling allows
data scientists to directly fix any errors that happen during or after
deployment.

How do you deploy new model versions?
> In addition, data scientists run experiments by using a custom-built A/B
testing service that allows them to define granular parameters. They then
analyze test results, and if they are deemed conclusive by the team, they
deploy the new version themselves.
>
> When it comes to deployment, we use a system similar to canary
development where we start by deploying the new version to one instance
and progressively update instances while monitoring performance. Data
scientists have access to a dashboard that shows the number of instances
under each version and continuous performance metrics as the deployment
progresses.
