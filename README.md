# Cell Counting in Microscopic Fluorescence

This repository contains a worked aimed at counting neuronal cells in fluorescent microscopic pictures of mouse brain slices.

A detail description, with code and results is presented in *notebooks/cell_counting_AML.ipynb*.

After that, the code has been reorganised into python module and additional notebooks have been created so to create small tutorials of each part of the analysis.

## Installation

Clone the repository where it is more convenient for you (it should get 336MB).

```r
git clone https://baltig.infn.it/clissa/cell_counting_AML

```

Once the download is finished, enter the cell_counting_AML directory and setup a conda environment using the requirements.txt.

```r
cd cell_counting_AML
conda create --name <env> --file requirements.txt

```

## Configure Jupyter Extensions

Now you have all is needed to run the notebooks. However, before starting it may be convenient to add some aesthetic and functional configurations for jupyter.
If you want to install these extensions run the following code. Otherwise skip to the next stage.

```r
cp custom.css ~/.jupyter/custom/ #Note: you may need to create the folder ~/.jupyter/custom/ before copying the configuration file
conda install -c conda-forge jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

#enable useful extensions
jupyter nbextension enable toc2/main
```
If you find any trouble with nb_extensions, please refer to these links for more details:

[Documentation](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html)

[Blog post](https://towardsdatascience.com/jupyter-notebook-extensions-517fa69d2231)

## Usage

Once you're ready, you can have a "guided tour" of the analysis reading the detailed notebook *notebooks/cell_counting_AML.ipynb* (Note: this is intended as a report and it is not supposed to work for any user).
Otherwise you can start to explore the analysis workflow through the tutorial notebooks. The suggested order is the following:

- *Data_Exploration.ipynb*
- *Pre-processing_and_augmentation.ipynb*
- *Trainer.ipynb*
- *Visualize_Results.ipynb*
- *Performance_Assessment.ipynb*

Also, take the chance to play with the python modules contained in the code/ folder and personalise according to your needs.

**Hope you will enjoy!**