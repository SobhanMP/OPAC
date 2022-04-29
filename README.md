# OPAC: Optimal Packet over Avian Carriers



The complete report is  [here](https://raw.githubusercontent.com/SobhanMP/OPAC/master/report/pigeon.pdf). 


It contains all of the detail, formulation, methodology, problem definition, simulation.
Long story short, we want to use pigeons to improve the efficiency of mesh networks (IP networks with loops). I defined the problem, simulated the data, and solved the problem with Benders decomposition and Dantzig-Wolfe decomposition.


The code can be ran by adding the packages via `import Pkg; Pkg.dev("https://github.com/SobhanMP/OPAC")` and then running the scripts in the order showed. The fist one generates the instances (and can be skipped) it can benefit from more processes ex. `-p 8`. The second one generates the table that compares the col-gen process and the third one generates the table comparing the problem solving speed.  they write their result to the standard output (only the last 2) and to files. They are resumable in the sense that re-running the scripts will cause them to continue their task. The last one generates the figures.


Developed in the context of Ivan Contreras's INDU 6361 Discrete Optimization.
