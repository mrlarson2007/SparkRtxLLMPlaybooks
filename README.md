# SparkRtxLLMPlaybooks
repro that allows you to setup your new spark rtx laptop using recipes from Nvidia.

## Quick start

[`rtx_spark_setup.ipynb`](rtx_spark_setup.ipynb) walks you through running a model from
[NVIDIA's Local AI Inference Recipes](https://nvidia.github.io/local-ai-inference-recipes/hardware/rtx-spark/index.html)
one step at a time. You enter your laptop specs (memory), the model and the engine. The notebook then:

1. Detects your OS, memory and GPU.
2. Downloads the recipe catalog. It keeps a cached copy for 24 hours.
3. Picks the recipe that fits your memory budget and prints its prerequisites, limitations and notes.
4. Installs the pinned llama.cpp build to `~/llama-cuda/<build>`. You can set `LLAMA_CPP_DIR` to install
   somewhere else, or `LLAMA_SERVER_PATH` to use an existing `llama-server`.
5. Downloads the model, draft/MTP and vision projector files to `MODELS_DIR`. The default is `~/llm-models`.
6. Builds the recipe's launch command using those local files and saves it as a `.ps1` script.
7. Starts `llama-server`, then 8. sends a test prompt to the OpenAI-compatible endpoint.

Each step checks whether it's already done (build installed, files present with the right size, server
healthy) and skips it if so, so you can re-run the notebook safely.

### Requirements

- Windows on an RTX Spark laptop, with the latest NVIDIA driver installed.
- Python 3.10+ with Jupyter (`pip install notebook`). `huggingface_hub` is installed automatically.
- Optional: set the `HF_TOKEN` environment variable for gated Hugging Face models. Set `GITHUB_TOKEN` to
  raise the GitHub API rate limit.

```powershell
pip install notebook
jupyter notebook rtx_spark_setup.ipynb
```

The helper logic is in [`rtx_spark_setup.py`](rtx_spark_setup.py). Run the tests with
`python -m unittest discover -s tests`.
