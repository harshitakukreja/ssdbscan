# Semi Supervised DBSCAN

This repository implements a Semi Supervised version of DBSCAN clustering algorithm in Python from the paper:
> [Semi-supervised Density-Based Clustering. _Levi Lelis and Jorg Sander. IEEE International Conference on Data Mining 2009_](https://webdocs.cs.ualberta.ca/~santanad/papers/2009/lelisS09.pdf)

<!-- Tell something about what this code does --> 

## Usage

### Example Usage
    >>> from ss_clustering import SSDBSCAN
    >>> import numpy as np
    >>> from sklearn.datasets import make_moons
    >>> X, _ = make(n_samples=200, noise=0.05, random_state=0)
    >>> y = np.full(X.shape[0], -1)
    >>> y[0] = 0
    >>> y[1] = 0
    >>> y[100] = 0
    >>> y[101] = 1
    >>> clustering = SSDBSCAN().fit(X, y)
    >>> clustering.labels_

<!-- ## Some results -->

## License

Copyright (c) 2025 Harshita Kukreja
For licence information, see [LICENSE](https://github.com/harshitakukreja/ssdbscan/blob/main/LICENSE)

---

For bugs in the code, please write to: harshitakukreja [at] nyu [dot] edu


