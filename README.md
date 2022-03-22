# graph-substructures

Repository with the code used for the paper "Bayesian Learning of Graph
Substructures" by Willem van den Boom, Maria De Iorio and Alexandros Beskos
(in preparation).


## Description of files

* [`graph_substructures.py`](graph_substructures.py) is a Python module that
provides an implementation of the MCMC algorithms for the Gaussian graphical
model with the degree-corrected stochastic blockmodel or the Southern Italian
community structure as prior distribution on the graph. It only covers the
single graph case. The scripts [`karate.py`](karate.py),
[`structure_recovery.py`](structure_recovery.py) and [`fund.py`](fund.py)
import it. It imports [`wwa.py`](wwa.py).

* [`wwa.py`](wwa.py) is a Python module that provides an implementation of the
*G*-Wishart weighted proposal algorithm
(WWA, van den Boom et al., 2022, [doi:10.1080/10618600.2022.2050250]). It uses
the C++ code in [`wwa.cpp`](wwa.cpp) via the header file [`wwa.h`](wwa.h).

* [`karate.py`](karate.py) produces the results for the simulation study with
the karate club network.

* [`structure_recovery.py`](structure_recovery.py) produces the figure for the
simulation study on block structure recovery.

* [`fund.py`](fund.py) produces the results for the application to mutual fund
data. It reads in the data from the text file
[`ExampleSection6.txt`](ExampleSection6.txt) which is available from the
[supplemental material] of
Scott & Carvalho (2008, [doi:10.1198/106186008X382683])

* [`gene.py`](gene.py) produces the figure for the application to the gene
expression data downloaded from The Cancer Genome Atlas by the R script
[`gene.R`](gene.R). [`gene.py`](gene.py) imports [`wwa.py`](wwa.py).

* [`environment.yml`](environment.yml) details the conda environment used for
the paper. It can be used to [recreate the environment]. The dependencies of
the C++ script [`wwa.cpp`](wwa.cpp) are detailed preceding the respective
include directives.


[doi:10.1080/10618600.2022.2050250]: https://doi.org/10.1080/10618600.2022.2050250
[supplemental material]: https://www.tandfonline.com/doi/suppl/10.1198/106186008X382683/suppl_file/ucgs_a_10711946_sm0001.zip
[doi:10.1198/106186008X382683]: https://doi.org/10.1198/106186008X382683
[recreate the environment]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
