# Welcome to ESMDAPY
esmdapy is a program

## Purpose

SPOTPY is a Python framework that enables the use of Computational optimization techniques for calibration, uncertainty and sensitivity analysis techniques of almost every (environmental-) model. The package is puplished in the open source journal PLoS One:

Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made Python Package, PLoS ONE, 10(12), e0145180, [doi:10.1371/journal.pone.0145180](http://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0145180 "doi:10.1371/journal.pone.0145180"), 2015
 
The simplicity and flexibility enables the use and test of different 
algorithms without the need of complex codes:

```
sampler = spotpy.algorithms.sceua(model_setup())     # Initialize your model with a setup file
sampler.sample(10000)                                # Run the model
results = sampler.getdata()                          # Load the results
spotpy.analyser.plot_parametertrace(results)         # Show the results
```


## Features
Complex algorithms bring complex tasks to link them with a model. 
We want to make this task as easy as possible. 
Some features you can use with the SPOTPY package are:
