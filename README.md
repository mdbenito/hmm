# hmm

This is a rather simple implementation of a discrete Hidden Markov Model 
with discrete emissions given either by probability tables or Poisson
distributions. Estimation of the parameters of the model (initial, transition
and emission probabilities) is done using the Baum-Welch algorithm for HMMs,
which is actually Expectation Maximization, a standard iterative algorithm
for maximum likelihood estimation.

The main insight won with this project is how badly EM fails to converge
to the true model parameters even in the simplest of cases no matter how big
the data set is, due to a tendency to get stuck in local optima.
Regularization by means of a prior on the parameters and maximum a posteriori
estimation should help but leaves us again with point estimates and possibly
local optima. Going fully Bayesian seems to be the way!

Another issue is the tendency of the alpha and beta quantities in the forward
and backward passes to reach zero. Because of the recursive nature of the
algorithm this implies that they remain zero thereafter, most likely away
from their true values. This too could be addressed by choosing a prior on the
parameters which were concentrated away from zero. *(TODO)*


## Disclaimer

This is neither a library nor an implementation ready for production. Although
there are unit tests which make me quite confident that computations are right
and the operations are vectorized it's still Python and consequently
depressingly slow. There are libraries implementing this algorithm with many
more features and optimization as well. You should probably use those.

* [hmmlearn](https://github.com/hmmlearn): hmmlearn is a set of algorithms
  for unsupervised learning and inference of Hidden Markov Models. The brunt
  of the computations is done by compiled cython code.

For supervised learning use:

* [seqlearn](https://github.com/larsmans/seqlearn): seqlearn is a sequence
  classification toolkit for Python. It is designed to extend scikit-learn
  and offer as similar as possible an API. It is also developed in cython.

## Dependencies

* Python >= 3.5
* Numpy
* matplotlib

You can use [conda](http://conda.pydata.org/) to install everything in a
breeze.

## Usage

The driver should be `main.py`, though this was intended to be used with
specific data and has not been finished. Only tests will run, using:

``python main.py``

There are usage examples in `test_inference.py`. In particular check the
method `TestMethods.test_iterate_simple()` in this file for a demo of the
issues that maximum likelihood encounters while trying to fit parameters for
a tiny model.

## Possible further work

There is preliminary work for the introduction of history effects. These can
be of two types:

* Past emissions influencing the current emission.
* Past emissions influencing the current state.

These two settings do not alter the Markov property of the state sequence, so
we can apply the same theory as before. *(check me)*

## References

[1] Rabiner, Lawrence R. “A Tutorial on Hidden Markov Models and Selected
Applications in Speech Recognition.” In Proceedings of the
IEEE, 77:257–86, 1989. doi:10.1.1.131.2084.

[2] Bishop, Christopher M. “Pattern Recognition and Machine Learning.” 2nd ed.
Information Science and Statistics. Springer, 2006. (Chapter 13).

[3] Mark Stamp: A revealing introduction to Hidden Markov Models.

[4] Escola, Sean, Alfredo Fontanini, Don Katz, and Liam Paninski.
 “Hidden Markov Models for the Stimulus-Response Relationships of Multistate
 Neural Systems.” Neural Computation 23, no. 5 (May 2011): 1071–1132.
 doi:10.1162/NECO\_a\_00118.


## License

This software falls under the GNU general public license version 3 or later.
It comes without **any warranty whatsoever**.
For details see http://www.gnu.org/licenses/gpl-3.0.html.

