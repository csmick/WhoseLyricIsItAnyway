# WhoseLyricIsItAnyway
Utilizes natural language processing techniques to guess the artist based on the lyrics of the song

## Getting Started

First, you want to clone this repository to your local machine using the following command:

```
git clone https://github.com/csmick/WhoseLyricIsItAnyway.git
```

### Prerequisites

In order to run the code for this project, you will need the following items

- python modules in requirements.txt
- GloVe word embeddings (specifically those of length 100)

### Installing

To install the python modules from the requirements document, use the command:

```
pip3 install -r requirements.txt
```

I would recommend creating a python virtual environment before running this command, but you can also install them systemwide if you like.

To install the GloVe word embedings, follow the installation instructions on the [GloVe webpage](https://nlp.stanford.edu/projects/glove) and ensure that glove.6B.100d.txt ends up in the main project directory.

## Running the code

### Naive Bayes

To run the code for the baseline Naive Bayes model, simply run the command:

```
python3 baseline_model.py
```

The output should simply be an accuracy percentage.

### Neural Network

To run the code for the Recurrent Neural Network model, simply run the command:

```
python3 nn.py
```

The output should look something like this:

![NeuralNetworkOutput](https://github.com/csmick/WhoseLyricIsItAnyway/blob/master/images/NeuralNetworkOutput.png "Neural Network sample output")

