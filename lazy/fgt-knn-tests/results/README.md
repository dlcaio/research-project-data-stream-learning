# Stream Learning Research Project

## [Experiments' Results](https://github.com/dlcaio/research-project-stream-learning/tree/master/lazy/fgt-knn-tests/results/)

#### Artificial
* [SEA Concepts](https://github.com/vlosing/driftDatasets/tree/master/artificial/sea)
	* 50000 instances
	* Drift Properties:
		* Abrupt
		* Real
	* 2 classes
	* 3 features
	![](rotating-hyperplane-3.png =50x20)
* [Rotating Hyperplane](https://github.com/vlosing/driftDatasets/tree/master/artificial/hyperplane)
	* 200000 instances
	* Drift Properties:
		* Incremental
		* Real
	* 2 classes
	* 10 features
* [Interchanging RBF](https://github.com/vlosing/driftDatasets/tree/master/artificial/rbf) **[1]**
	* 200000 instances
	* Drift Properties:
		* Abrupt
		* Real
	* 2 classes
	* 10 features
* [Transient Chessboard](https://github.com/vlosing/driftDatasets/tree/master/artificial/chess) **[2]**
	* 200000 instances
	* Drift Properties:
		* Abrupt
		* Reoccurring
		* Virtual
	* 8 classes
	* 2 features
* [Mixed Drift](https://github.com/vlosing/driftDatasets/tree/master/artificial/mixedDrift) **[3]**
	* 600000 instances
	* Drift Properties:
		* Various
			* Real
			* Virtual
	* 15 classes
	* 2 features
#### Real World
* [Weather](https://github.com/vlosing/driftDatasets/tree/master/realWorld/weather) ([original source](http://users.rowan.edu/~polikar/research/nse/))
	* 18159 instances
	* Drift Properties:
		* Virtual
	* 2 classes
	* 8 features
* [Poker Hand](https://github.com/vlosing/driftDatasets/tree/master/realWorld/poker) ([original source](https://archive.ics.uci.edu/ml/datasets/Poker+Hand))
	* 829201 instances
	* Drift Properties:
		* Virtual
	* 10 classes
	* 10 features




## References
**1, 2, 3.** V. Losing, B. Hammer and H. Wersing, "KNN Classifier with Self Adjusting Memory for Heterogeneous Concept Drift," 2016 IEEE 16th International Conference on Data Mining (ICDM), Barcelona, 2016, pp. 291-300, doi: 10.1109/ICDM.2016.0040.
