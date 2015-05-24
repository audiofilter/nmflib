function [W,H,errs,vout] = nmf_euc_orth(V, r, varargin)
% function [W,H,errs,vout] = nmf_euc_orth(V, r, varargin)
%
% Implements Orthogonal NMF using squared error cost function (see [1-2] for details):
% 
%      min sum(sum((V-W*H).^2)) s.t. W>=0,H>=0
%
% Inputs: (all except V and r are optional and passed in in name-value pairs)
%   V      [mat]  - Input matrix (n x m)
%   r      [num]  - Rank of the decomposition
%   niter  [num]  - Max number of iterations to use [100]
%   thresh [num]  - Number between 0 and 1 used to determine convergence;
%                   the algorithm has considered to have converged when:
%                   (err(t-1)-err(t))/(err(1)-err(t)) < thresh
%                   ignored if thesh is empty [[]]
%   norm_w [num]  - Type of normalization to use for columns of W [1]
%                   can be 0 (none), 1 (1-norm), or 2 (2-norm)
%   norm_h [num]  - Type of normalization to use for rows of H [0]
%                   can be 0 (none), 1 (1-norm), 2 (2-norm), or 'a' (sum(H(:))=1)
%   verb   [num]  - Verbosity level (0-3, 0 means silent) [1]
%   W0     [mat]  - Initial W values (n x r) [[]]
%                   empty means initialize randomly
%   H0     [mat]  - Initial H values (r x m) [[]]
%                   empty means initialize randomly
%   W      [mat]  - Fixed value of W (n x r) [[]] 
%                   empty means we should update W at each iteration while
%                   passing in a matrix means that W will be fixed
%   H      [mat]  - Fixed value of H (r x m) [[]] 
%                   empty means we should update H at each iteration while
%                   passing in a matrix means that H will be fixed
%   myeps  [num]  - Small value to add to denominator of updates [1e-20]
%   orth_w [bool] - Switch that determines whether we are imposing orthogonality 
%                   constraints on W or not. [true] 
%   orth_h [bool] - Switch that determines whether we are imposing orthogonality 
%                   constraints on H or not. [false] 
%
% Outputs:
%   W      [mat]  - Basis matrix (n x r)
%   H      [mat]  - Weight matrix (r x m)
%   errs   [vec]  - Error of each iteration of the algorithm
%
% [1] S. Choi, "Algorithms for Orthogonal Nonnegative Matrix Factorization",
%     IEEE International Joint Conference on Neural Networks, 2008.
% [2] D. Lee and S. Seung, "Algorithms for Non-negative Matrix Factorization", 
%     NIPS, 2001.
%
% 2010-11-04 Graham Grindlay (grindlay@ee.columbia.edu)

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

% do some sanity checks
if min(min(V)) < 0
    error('Matrix entries can not be negative');
end
if min(sum(V,2)) == 0
    error('Not all entries in a row can be zero');
end

[n,m] = size(V);

% process arguments
[niter, thresh, verb, myeps, norm_w, norm_h, orth_w, orth_h, W0, H0, W, H] = ...
    parse_opt(varargin, 'niter', 100, 'thresh', [], 'verb', 1, 'myeps', 1e-20, ...
                        'norm_w', 1, 'norm_h', 0, 'orth_w', 1, 'orth_h', 0, ...
                        'W0', [], 'H0', [], 'W', [], 'H', []);

% initialize W based on what we got passed
if isempty(W)
    if isempty(W0)
        W = rand(n,r);
    else
        W = W0;
    end
    update_W = true;
else 
    update_W = false;
end

% initialize H based on what we got passed
if isempty(H)
    if isempty(H0)
        H = rand(r,m);
    else
        H = H0;
    end
    update_H = true;
else % we aren't H
    update_H = false;
end

if ~(norm_w && orth_w) && ~(norm_h && orth_h)
    warning('nmf_euc_orth: orthogonality constraints should be used with normalization on the same mode!');
end

if norm_w ~= 0
    % normalize W
    W = normalize_W(W,norm_w);
end

if norm_h ~= 0
    % normalize H
    H = normalize_H(H,norm_h);
end

errs = zeros(niter,1);
for t = 1:niter
    % update W if requested
    if update_W
        if orth_w
            W = W .* ( (V*H') ./ max(W*H*V'*W, myeps) );
        else
            W = W .* ( (V*H') ./ max(W*(H*H'), myeps) );
        end
        if norm_w ~= 0
            W = normalize_W(W,norm_w);
        end
    end
    
    % update H if requested
    if update_H
        if orth_h
            H = H .* ( (W'*V) ./ max(H*V'*W*H, myeps) );
        else
            H = H .* ( (W'*V) ./ max((W'*W)*H, myeps) );
        end
        if norm_h ~= 0
            H = normalize_H(H,norm_h);
        end
    end
    
    % compute squared error
    errs(t) = sum(sum((V-W*H).^2));
    
    % display error if asked
    if verb >= 3
        disp(['nmf_euc_orth: iter=' num2str(t) ', err=' num2str(errs(t))]);
    end
    
    % check for convergence if asked
    if ~isempty(thresh)
        if t > 2
            if (errs(t-1)-errs(t))/(errs(1)-errs(t-1)) < thresh
                break;
            end
        end
    end
end

% display error if asked
if verb >= 2
    disp(['nmf_euc: final_err=' num2str(errs(t))]);
end

% if we broke early, get rid of extra 0s in the errs vector
errs = errs(1:t);

% needed to conform to function signature required by nmf_alg
vout = {};
