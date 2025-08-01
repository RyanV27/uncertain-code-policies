# Code as Policies

## Cloning the Repository

Clone this repository.

```
git clone https://github.com/RyanV27/code-as-policies.git
```

Change to the project directory.

```
cd code-as-policies
```

## Python Environment Setup

Create a Python virtual environment with Python version `3.8`. This is the version that I used.

Install the required packages.

```
pip install -r requirements.txt
```

## Running the Experiment

To start the experiment, run the code below.

```
python Interactive_Demo.py
```

Images and videos of the experiment will be stored in the `./runs` directory. Once you run the code, this directory should automatically get created with a subdirectory containing the files for the current run. You can use these as a reference when giving tasks to perform. 

Write down the task to perform when prompted with `User input: ` in the terminal. For each given task, the corresponding video should be generated in the `./runs` directory.

## Changing the Experiment Setup

In case you want to modify the setup, you'll most likely only need to change the variables in the `interactive_demo.py` file.

Variables in `interactive_demo.py`:-
- num_blocks: Sets the number of blocks to generate.
- num_bowls: Sets the number of blocks to generate.
- model_id: Name of the Hugging Face model. (If you are changing this, then it's better to change `model_name` also in the `tabletop_config.py` file.)
- cache_dir: This is a parameter in AutoTokenizer.from_pretrained() and AutoModelForCausalLM.from_pretrained() used to specify the location where you want to store the model. If it is left empty, then the model will be stored in the `.cache` directory.