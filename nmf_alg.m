function [W,H,errs,varout] = nmf_alg(V,r,varargin)
% function [W,H,errs,varout] = nmf_alg(V,r,varargin)
%
% Perform Non-negative Matrix Factorization of V into W*H s.t. W,H >= 0
% Aside from V and r, all inputs are passed in name-value pairs (e.g to
% pass @nmf_kl for the alg parameter you would include 'alg' followed 
% by @nmf_alg in the input arguments.
%
% Common inputs (other args for specific algorithms can also be passed in):
%   V       [mat]  - Input data matrix (n x m)
%   r       [num]  - Rank of the decomposition
%   alg     [fcn]  - Function handle of specific NMF algorithm to use. 
%                    Functions must conform to the following signature:
%                    [W,H,errs,varargout] = f(V,r,varargin); [@nmf_kl]
%   nrep    [num]  - Number of repetitions to try. [1]
%   seedrep [bool] - Use seed-based repitions?   [false]
%                    If true, only space for a single W and single H are
%                    kept around as the random seed is stored and used to
%                    reproduce the best run.  This means that nrep+1 runs
%                    of the algorithm are run.  If false, the results of
%                    each run are kept and those from the best run are
%                    returned at the end.  This is faster (only nrep
%                    algorithm runs), but uses more memory
%                    (nrep*numel(W) + nrep*numel(H)).
%   verb    [num]  - Verbosity level (0-3, 0 means silent) [1]
% (the following inputs are handled by the specific algorithm called, but 
%  all are common inputs)
%   niter   [num]  - Max number of iterations to use [100]
%   thresh  [num]  - Number between 0 and 1 used to determine convergence;
%                    the algorithm has considered to have converged when:
%                    (err(t-1)-err(t))/(err(1)-err(t)) < thresh
%                    ignored if thesh is empty [[]]
%   norm_w  [str]  - Type of normalization to use for columns of W [1]
%                    can be 0 (none), 1 (1-norm), or 2 (2-norm)
%   norm_h  [str]  - Type of normalization to use for rows of H [0]
%                    can be 0 (none), 1 (1-norm), 2 (2-norm), or 'a' (sum(H(:))=1)
%   W0      [mat]  - Initial W values (n x r) [[]]
%                    empty means initialize randomly
%   H0      [mat]  - Initial H values (r x m) [[]]
%                    empty means initialize randomly
%   W       [mat]  - Fixed value of W (n x r) [[]]
%                    empty means we should update W at each iteration while
%                    passing in a matrix means that W will be fixed
%   H       [mat]  - Fixed value of H (r x m) [[]]
%                    empty means we should update H at each iteration while
%                    passing in a matrix means that H will be fixed
%   myeps   [num]  - Small value to add to denominator of updates [1e-20]
%
% Outputs:
%   W       [mat]  - Basis matrix (n x r)
%   H       [mat]  - Weight matrix (r x m)
%   errs    [vec]  - Error of each iteration of the algorithm
%   varout  [cell] - cell array of additional outputs returned by NMF algs
%
% 2010-01-14 Graham Grindlay (grindlay@ee.columbia.edu)

% Copyright (C) 2008-2028 Graham Grindlay (grindlay@ee.columbia.edu)
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

% process args needed here, ignore the rest
[alg,nrep,seedrep,verb] = parse_opt(varargin, 'alg', @nmf_kl, 'nrep', 1, ...
                                              'seedrep', false, 'verb', 1);

% only worry about rep-related stuff if we're doing more than 1 rep
if nrep > 1
    % if not seed-based reps, make some space to hold the results
    if ~seedrep
        cargout = cell(nrep,1);
    end
    
    seed = rand('seed');
    dnorm = zeros(1,nrep);
    
    for i = 1:nrep
        % set seed so we can reproduce run
        rand('seed', seed+i);
        
        % run the algorithm being used
        [W,H,errs,vout] = alg(V,r,varargin{:});

        % store this run's results if we're not doing seed-based reps
        if ~seedrep
            cargout{i} = {W,H,errs,vout};
        end
        dnorm(i) = errs(end);
        
        % display rep error if asked
        if verb >= 2
            disp(['nmf_alg: rep=' num2str(i) ', err=' num2str(errs(end))]);
        end
    end
    
    % sort error norms
    [junk, ndx] = sort(dnorm, 'ascend');
    
    if ~seedrep
        W = cargout{ndx(1)}{1};
        H = cargout{ndx(1)}{2};
        errs = cargout{ndx(1)}{3};
        varout = cargout{ndx(1)}{4};
        
        % display final error if asked
        if verb >= 1
            disp(['nmf_alg: final err=' num2str(errs(end))]);
        end
            
        return;
    else
        % recompute the run with the lowest error norm
        rand('seed', seed+ndx(1));
    end
end

% run the algorithm being used
[W,H,errs,varout] = alg(V,r,varargin{:});

% display final error if asked
if verb >= 1
    disp(['nmf_alg: final err=' num2str(errs(end))]);
end
