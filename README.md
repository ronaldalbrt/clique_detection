<h1 align="center">
<br> Clique detection in Graphs with the usage of Graph Neural Networks (GNNs)
</h1>
Repository for the course on <a href="https://sites.google.com/poli.ufrj.br/jsouza/port/pee/aprendizado-de-m%C3%A1quina-em-grafos">Machine Learning in Graphs</a> at  <a href="https://www.cos.ufrj.br/index.php/pt-BR/" > PESC - Programa de Engenharia de Sistemas e Computação</a> from <a href="https://ufrj.br/" >UFRJ - Federal University of Rio de Janeiro</a>, taught by <a href="https://www.cos.ufrj.br/~gerson/">Prof. Gerson Zaverucha</a>.

Developed by Ronald Albert.
<h2 align="center">
The project
</h2>
The project is an implementation of Learning on Graphs techniques in the well known problem of Clique Detection. The project consists on modeling the clique detection as a node classification problems, where nodes that belong to a clique are classified as 1, and nodes not belonging to a clique are classified as 0.

It's entirely implemented in python and requires several python libraries to be executed. All of the required libraries are listed at [requirements.txt](requirements.txt) as well as their respective versions. In order to install all the necessary package one could run the following command
```
pip install -r requirements.txt
```

<h2 align="center">
File list
</h2>
<ul>
    <li><h3>run.py</h3></li>
    <p>Script that runs the entire project. It's the main file of the project, and generates the results available at the results folder.</p>
    <li><h3>experiment.py</h3></li>
    <p>Script where a single experiment is defined. It estimates and evaluates a model on the train, validation and test sets</p>
    <li><h3>dataset.py</h3></li>
    <p> Script where the dataset is defined. It's responsible for loading the dataset and generating the train, validation and test sets.</p>
    <li><h3>eval.py</h3></li>
    <p> Script where the evaluation of the model is defined. It's responsible for calculating the metrics of the model on the train, validation and test sets.</p>
    <li><h3>models/</h3></li>
    <p>Folder where each of the implemented models are defined. All of them are implemented with the PyTorch Geometric package from python.</p>
</ul>

<h2 align="center">
Execution
</h2>
<p>After installing all the necessary packages, one can run the project by executing the following command</p>

```
python run.py
```

<h2 align="center">
Results
</h2>
<p>The results obtained by running the project are available at the results folder. The results are in pickle format and can de loaded into a python Dictionary object by perfoming the command</p>

```
with open('results/<result_file_name>.pkl', 'rb') as f:
    results = pickle.load(f)
```
