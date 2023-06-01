This project contains the code I used for my Bachelor's thesis in Engineering Physics, I'll add a link to it when it's uploaded.
It's all a big mess, and was only intended for this specific project.
But it can absolutely be used as a reference or a starting point for other studies of similar things.

What other people may find especially useful is `D2_Variable.py`, which contains first- and second derivative summation by parts operators
with variable coefficients from "Summation by Parts Operators for Finite Difference Approximations of Second-Derivatives with Variable Coefficients" 
by Ken Mattson [*Journal of Scientific Computing* 51.3 (2012), s. 650–682], implemented in 2D.

For my actual thesis, the model is in `var_b.py`, which is very flexible and can accept any general A and B.
`error.py` contains the code for my actual results.
`ref.py` handles importing of reference solutions from Matlab. 
`plotting.py` plots data. Generation of the actual figures used in the report is in `figures.py` and `complicated.py`.
The rest are either for testing or are various utility files.
