# Actionable counterfactuals explanations using Bayesian networks
Code necessary to find data agnostic (exogenous and not leveraging on training data) actionable counterfactual explanations. Our approach is based on path planninh, sepcifically in potential fields theory and optimization using genetic algorithms. The code is structured in two main folders

# bayesace module
Main folder containing most of the actual functionality. 

The folder algorithms contains the different procedures implemented for finding counterfactual, namely the Wachter's algorithm, FACE and DAACE/BayesACE, all with a common interfacte (algorithm.py)

"models" contain the different conditional density estimators used, sharing as well a common interface. The most notable ones are RealNVP, a conditional linear Gaussian network and conditional KDE.

# experiments folder
Necessary code to reproduce the experiments carried out .

NOTE: Some experiments referring the density estimation and counterfactual explanations may fail, given that a custom modified version of PyBnesian is being used (found at https://github.com/Enrique-Val/PyBNesian). I plan to update the code to bypass this weird requirement, as a the major PyBnesian bug leading to making my own implementation has been fixed in a recent update.


