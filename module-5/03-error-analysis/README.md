# Error analysis

## Aims

Learning to analyze and understand the prediction errors of the trained model.

## Objectives

This is a short session where we will
- Learn to use Python APIs for model inference of DP
- Compute and analyze the root mean square error
- Make parity plots and error distribution plots

## Running Jupyter Notebook

The example code is given in the jupyter notebook ```Analysis.ipynb```. You can run it using:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/03-error-analysis/Analysis.ipynb)

## Computing the Root Mean Square Error

The most imporatant thing to check for a model is the root mean square error. 
For a quantity $X$ with predictions $X_{\text{pred}}$ and ground truth $X_{\text{true}}$, it is given by

$$ RMSE = \langle (X_{\text{pred}} - X_{\text{true}})^2 \rangle^{\frac{1}{2}}. $$

In DP, the training loss is a combination of the mean square error of energy, force, and sometimes virial.
The error for energy and force is usually no more than 1 meV/atom and $10^{-1}$ eV/Å, 
with the specifics depending on the system. The model and training hyperparameters can also have an influence, 
see for example [model size](https://github.com/deepmodeling/deepmd-kit/blob/master/doc/troubleshooting/howtoset_netsize.md). 

For typical machine learning tasks where data are drawn from a single distribution, 
you should test the generalization error on a held-out dataset, 
which should not be too large compared to the training error.
However, here we use DP-GEN which gives a series of different distributions, 
so you may check the error on the constituent data systems in the jupyter notebook, 
which is of more interest.

## Parity Plots

Also a common type of analysis is parity plots, 
which is a scatterplot that compares the predicted value against the true value. 
A well-trained model's predictions should not display significant outliers. 
We've plotted the one for the force, and you can try doing the same for the energy.

## Error Distribution

We can also plot the error distribution. Usually they should give a Gaussian-like distribution, but our dataset has several different constituents, which may make things slightly more complicated. Again, we plot the forces, and you can do the same for the energy, and you can also try it on the different subsets of data.
