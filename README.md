NMFlib v0.1.3
Graham Grindlay
grindlay@ee.columbia.edu
02/12/2010


===========================
======== Contents =========
===========================

This library contains implementations of a number of popular variants of the Non-negative Matrix Factorization (NMF) 
algorithm.  Currently, the library contains the following algorithms:

[nmf_alg]           - The primary wrapper function that all variants can be called from.  The wrapper's main responsibly
                      is to handle things like multiple restarts as well as serve as a comon entry point for all 
                      the NMF varieties listed below, although you certainly don't need to use it if you don't want to.

[nmf_kl]            - The original multiplicative update algorithm presented by Lee and Seung [1] that uses the normalized 
                      KL-divergence or I-divergence as its object function:  min sum(sum(V||W*H))
 
[nmf_euc]           - The original multiplicative update algorithm presented by Lee and Seung [1] that uses the Euclidean 
                      distance for its objective function:  min sum(sum((V-W*H).^2)) s.t. W>=0,H>=0

[nmf_kl_loc]        - Local NMF [2], a variant of the I-divergence objective that includes additional penalty terms that 
                      encourage  bases to be compact and orthogonal.

[nmf_kl_sparse_v]   - Variant of I-divergence NMF algorithm that includes a weighted penalty term to encourage sparsity 
                      in H [3].

[nmf_kl_sparse_es]  - Variant of the I-divergence NMF algorithm  that includes a weighted penalty term to encourage 
                      sparsity in H [4,5].

[nmf_euc_sparse_es] - Variant of the Euclidean NMF algorithm that includes a weighted penalty term to encourage sparsity 
                      H [4,6].

[nmf_kl_con]        - Implements Convolutive NMF [7] where each basis vector can now be a patch.

[nmf_amari]         - Uses the Amari alpha divergence [8] as the basis of the objective function to be minimized.  
                      Different values of alpha yield different divergences such as Pearson's distance (alpha=2), 
                      Hellinger's distance (alpha=0.5), and Neyman's chi-square distance (alpha=-1).

[nmf_beta]          - Uses the beta divergence [8] as the basis of the objective function to be minimized.  This gives 
                      difference functions for different values of beta: Itakura-Saito (beta=0), I-divergence (beta=1), 
                      Euclidean distance (beta=2).

[nmf_convex]        - Implements Convex NMF [9] which tries to find a factorization that minimizes sum(sum((V-V*W*H).^2)).
                      This leads to an interesting analog to k-means clustering where V*W represents cluster centroids
		      and H encodes the membership of each column of V in each centroid.  In contrast to the other NMF
		      algorithms, V can be of mixed sign in Convex NMF.

[nmf_euc_orth]      - Implements Orthogonal NMF [11] which tries to keep either the basis or weight vectors as orthogonal
                      as possible while still accurately reconstructing the data.

---------------------------
------- Utilities ---------
---------------------------

[normalize_W]       - Normalizes (typically) the columns of the W matrix (several types of normalization are supported).

[normalize_H]       - Normalizes the H matrix (several types of normalization are supported).

[parse_opt]         - Function to parse name-value argument pairs.  This is very much just a simplified version of Mark 
                      Paskin's process_options function, but does less error checking so its a little bit faster.

---------------------------
---------- Other ----------
---------------------------

[demo1]             - Script that shows the basic use of some of the algorithms on simulated data.

[demo2]             - Script that shows how to do basic music transcription using NMF.


===========================
=========== Usage =========
===========================

The function 'nmf_alg' is the primary entry point to the library.  You can access all NMF variants with this single 
function.  Its signature is as follows:

[W,H,errs,varout] = nmf_alg(V,r,varargin);

For example, to do 500 iterations of rank-5 KL-divergence NMF on a 100x500 element data matrix V, you would call the 
library as follows:

[W,H,err] = nmf_alg(V, 5, 'alg', @nmf_kl, 'niter', 500);

W will be a 100x5 matrix, H will be a 5x500 matrix, and err will be a 1x500 vector of error values.  Now, suppose we want 
to instead use Virtanen's [03] sparse NMF algorithm with a sparsity penalty of 0.1, but we want to run 3 repetitions of the 
algorithm (returning the best) to ameliorate the effects of random initialization.  In this case, we would do:

[W,H,err] = nmf_alg(V, 5, 'alg', @nmf_kl_sparse_v, 'niter', 500, 'nreps', 3, 'alpha', 0.1, 'verb', 1);

Note that in this case we have also provided a verbosity level (1) which will cause the library to print only the best
repetition's final (last iteration) error value.  As a final example, suppose we already have W and we want to just solve
for H.  This is done as follows:

[W,H,err] = nmf_alg(V, 5, 'alg', @nmf_euc, 'W', W);

Obviously, the W returned by the function call will be the same as the one we passed in.  Also, since we did not specify 
the number of iterations, the default value (100) would be used.

Note that it is possible to call the specific NMF variants directly, although you lose the ability to do multiple 
repetitions.  For example, to call the KL-divergence variant directly, you could do:

[W,H,err] = nmf_kl(V, 5, 'niter', 500, 'verb', 2);

---------------------------
--- Standard Parameters ---
---------------------------

The documentation throughout the library uses a standard argument specification format.  The name of the argument is
followed by its type in brackets.  The types are as follows:

   [num]  - A single scalar number
   [vec]  - A vector of scalars
   [mat]  - A matrix of scalars
   [cell] - A cell array
   [str]  - A string
   [bool] - A boolean value
   [fcn]  - A function handle
   [stct] - A struct

An argument description (which includes any size information) comes after the type and this is followed by the default
value for the argument in square brackets.  
 
The following parameters are always passed (in order) to nmf_alg directly):

   V       [mat]  - Input data matrix (n x m)
   r       [num]  - Rank of the decomposition

The rest of the parameters are optional (they have default values) and are passed in name-value pairs:

   alg     [fcn]  - Function handle of specific NMF algorithm to use. [@nmf_kl]
                    Functions must conform to the following signature: [W,H,errs,varargout] = f(V,r,varargin);
   nrep    [num]  - Number of repetitions to try [1]
   seedrep [bool] - Use seed-based repitions? [false]
                    If true, only space for a single W and single H are kept around as the random seed is stored and used 
                    to reproduce the best run.  This means that nrep+1 runs of the algorithm are run.  If false, the 
                    results of each run are kept and those from the best run are returned at the end.  This is faster 
                    (only nrep algorithm runs), but uses more memory (nrep*numel(W) + nrep*numel(H)).
   verb    [num]  - Verbosity level (0-3, 0 means silent) [1]
   niter   [num]  - Max number of iterations to use [100]
   thresh  [num]  - Number between 0 and 1 used to determine convergence. [[]] 
                    The algorithm has considered to have converged when: (err(t-1)-err(t))/(err(1)-err(t)) < thresh
                    Ignored if thesh is empty 
   norm_w  [str]  - Type of normalization to use for columns of W [1]
                    Can be 0 (none), 1 (1-norm), or 2 (2-norm)
   norm_h  [str]  - Type of normalization to use for rows of H [0]
                    Can be 0 (none), 1 (1-norm), 2 (2-norm), or 'a' (sum(H(:))=1)
   W0      [mat]  - Initial W values (n x r) [[]]
                    Empty means initialize randomly
   H0      [mat]  - Initial H values (r x m) [[]]
                    Empty means initialize randomly
   W       [mat]  - Fixed value of W (n x r) [[]]
                    Empty means we should update W at each iteration while passing in a matrix means that W will be fixed.
   H       [mat]  - Fixed value of H (r x m) [[]]
                    empty means we should update H at each iteration while passing in a matrix means that H will be fixed.
   myeps   [num]  - Small value to add to denominator of updates [1e-20]

These are the standard outputs:
 
   W       [mat]  - Basis matrix (n x r)
   H       [mat]  - Weight matrix (r x m)
   errs    [vec]  - Error of each iteration of the algorithm
   varout  [cell] - cell array of additional algorithm-specific outputs


===========================
========== Design =========
===========================

The NMFlib libary was designed with three principles in mind: correctness, speed, and consistency.  I have tried to 
keep the function signatures of each variant as consistent as possible so that it is relatively easy to experiment 
with different algorithms.  Since most of the algorithms support many optional parameters, all non-essential arguments 
(things other than the data V and the rank r) are passed in name-value pairs (see the demo scripts and examples above).
This means that you don't have to worry about remembering any kind of argument order.  

The code is *NOT* object oriented as early experiments suggested that speed and efficiency would suffer. However, the 
code is reasonably well documented and should therefore be easy to understand and extend.  Seeing as this project is 
very much a work-in-progress, I would very much appreciate any suggestions for improvement and/or bug fixes that you 
may have.  I will continue to add new algorithms as I find the time (there are several in my research pipeline now that 
should be making their way into v0.2 soon). Also, please drop me an email and let me know if you find the library useful 
for your work.  I'd love to hear about what you're working on and how the code has helped (or perhaps hindered!) you.

Thanks,
Graham Grindlay
02/12/2010
grindlay@ee.columbia.edu
http://www.ee.columbia.edu/~grindlay


===========================
======= References ========
===========================

[01] Lee, D. and Seung, S., "Algorithms for Non-negative Matrix Factorization", NIPS, 2001

[02] Li, S. et al., "Learning Spatially Localized, Parts-Based Representation", CVPR, 2001

[03] Virtanen, T. "Monaural Sound Source Separation by Non-Negative Factorization with Temporal Continuity and 
     Sparseness Criteria", IEEE Transactions on Audio, Speech, and Language Processing, vol. 15(3), 2007.

[04] Eggert, J. and Korner, E., "Sparse Coding and NMF", in Neural Networks, 2004

[05] Schmidt, M. "Speech Separation using Non-negative Features and Sparse Non-negative Matrix Factorization", 
     Tech. Report, 2007

[06] Schmidt, M. and Larsen, J. and Hsiao, F., "Wind Noise Reduction using Non-negative Sparse Coding", IEEE MLSP, 2007.

[07] Smaragdis, P., "Non-negative Matrix Factor Deconvolution; Extraction of Multiple Sound Sources from Monophonic Inputs",
     International Symposium on ICA and BSS, 2004.

[08] Cichocki, A. and Amari, S.I. and Zdunek, R. and Kompass, R. and Hori, G. and He, Z. "Extended SMART Algorithms 
     for Non-negative Matrix Factorization", Artificial Intelligence and Soft Computing, 2006

[09] Ding, C. and Li, T. and Jordan, M., "Convex and Semi-Nonnegative Matrix Factorizations", IEEE Transactions on 
     Pattern Analysis and Machine Intelligence, Vol. 99(1), 2008.

[10] Grindlay, G. and Ellis, D.P.W., "Multi-Voice Polyphonic Music Transcription Using Eigeninstruments", IEEE Workshop 
     on Applications of Signal Processing to Audio and Acoustics, 2009.

[11] S. Choi, "Algorithms for Orthogonal Nonnegative Matrix Factorization", IEEE International Joint Conference on 
     Neural Networks, 2008.

===========================
======== Changelog ========
===========================

03/11/2010 - v0.1 : Initial release

04/09/2010 - v0.1.1 : Fixed bug in nmf_convex.m that prevented proper handling of mixed-sign case (thanks to Charles Martin)

11/04/2010 - v0.1.2 : Fixed bugs in nmf_euc_sparse_es.m and nmf_kl_sparse_es.m that caused the wrong update equations to be used with
                      L1 normalization on W.  Also added some error checking to these functions.

11/05/2010 - v0.1.3 : Added support for Orthogonal NMF.