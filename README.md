# multilabel-dataset-resampling-algorithms
Set of algorithms used to resample (oversample and undersample) multilabel datasets.

All the algorithms return the list of indexes from the original dataset that you should either clone or remove (depending the algorithm).
None of these implementations modify the original dataset.

Implementations based on the paper:

Francisco Charte, Antonio J. Rivera, María J. del Jesus, Francisco Herrera,
Addressing imbalance in multilabel classification: Measures and random resampling algorithms,
Neurocomputing,
Volume 163,
2015,
Pages 3-16,
ISSN 0925-2312,
https://doi.org/10.1016/j.neucom.2014.08.091.
(https://www.sciencedirect.com/science/article/pii/S0925231215004269)


There are 2 implementations of MLSMOTE algorithm.

The original is the MLSMOTE.py, implemented exactly as described in the original paper. It returns the synthetic features and labels generated.

The modification of the algorithm is the file MLSMOTE_mod.py. It generates new samples until the minority label IR reaches the mean IR.

Implementations based on the paper:

Francisco Charte, Antonio J. Rivera, María J. del Jesus, Francisco Herrera,
MLSMOTE: Approaching imbalanced multilabel learning through synthetic instance generation,
Knowledge-Based Systems,
Volume 89,
2015,
Pages 385-397,
ISSN 0950-7051,
https://doi.org/10.1016/j.knosys.2015.07.019.
(https://www.sciencedirect.com/science/article/pii/S0950705115002737)
