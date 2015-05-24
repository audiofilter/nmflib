function [W,H,errs,vout] = nmf_convex(V, r, varargin)
% function [W,H,errs,vout] = nmf_convex(V, r, varargin)
%
% Implements Convex NMF (see [1] for details) where each column of V*W 
% represents one of r cluster centroids, each of which is a convex 
% combination of the columns of V (i.e. W gives the weight of each data 
% column for each cluster). Each row of H gives the weighting of each 
% cluster centroid for each data column vector.  
% 
%      min sum(sum((V-V*W*H).^2)) s.t. W>=0,H>=0
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
%   norm_h [num]  - Type of normalization to use for rows of H [1]
%                   can be 0 (none), 1 (1-norm), 2 (2-norm), or 'a' (sum(H(:))=1)
%   verb   [num]  - Verbosity level (0-3, 0 means silent) [1]
%   W0     [mat]  - Initial W values (m x r) [[]]
%                   empty means initialize randomly
%   H0     [mat]  - Initial H values (r x m) [[]]
%                   empty means initialize randomly
%   W      [mat]  - Fixed value of W (m x r) [[]] 
%                   empty means we should update W at each iteration while
%                   passing in a matrix means that W will be fixed
%   H      [mat]  - Fixed value of H (r x m) [[]] 
%                   empty means we should update H at each iteration while
%                   passing in a matrix means that H will be fixed
%   myeps  [num]  - Small value to add to denominator of updates [1e-20]
%
% Outputs:
%   W      [mat]  - Centroid matrix (m x r)
%   H      [mat]  - Cluster matrix (r x m)
%   errs   [vec]  - Error of each iteration of the algorithm
%
% [1] D. Lee and S. Seung, "Algorithms for Non-negative Matrix Factorization", 
%     NIPS, 2001
% [2] C. Ding and T. Li and M. Jordan, "Convex and Semi-Nonnegative Matrix
%     Factorizations", IEEE Transactions on Pattern Analysis and Machine
%     Intelligence, Vol. 99(1), 2008.
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

% do some sanity checks
if min(sum(V,2)) == 0
    error('Not all entries in a row can be zero');
end

m = size(V,2);

% process arguments
[niter, thresh, norm_w, norm_h, verb, myeps, W0, H0, W, H] = ...
    parse_opt(varargin, 'niter', 100, 'thresh', [], 'norm_w', 1, ...
                        'norm_h', 0, 'verb', 1, 'myeps', 1e-20, ...
                        'W0', [], 'H0', [], 'W', [], 'H', []);

% initialize W based on what we got passed
if isempty(W)
    if isempty(W0)
        % if non-negative datam, use NMF-EUC to do init. as suggested in [2]
        if min(V(:)>=0)
            W = nmf_euc(V', r, 'niter', 100);
            W = W + 0.2*ones(size(W));
        else % just do random initialization
            % NOTE: it's probably much better to do a k-mean initialization
            W = rand(m,r);
        end
    else
        W = W0;
    end
    update_W = true;
else 
    update_W = false;
end

if norm_w ~= 0
    % normalize W
    W = normalize_W(W,norm_w);
end

% initialize H based on what we got passed
if isempty(H)
    if isempty(H0)
        % do initialization for H as suggested in [2]
        H0 = (W/(W'*W))';
        H = zeros(size(H0));
        H(H0>0) = H0(H0>=0)+0.2*ones(size(H0(H0>=0)))*sum(H0(H0>=0))./sum(H0(:)>0);
        H = max(H,myeps);
    else
        H = H0;
    end
    update_H = true;
else % we aren't H
    update_H = false;
end

if norm_h ~= 0
    % normalize H
    H = normalize_W(H,norm_h);
end

Yp = 0.5*(abs(V'*V) + (V'*V));
Yn = 0.5*(abs(V'*V) - (V'*V));

errs = zeros(niter,1);
for t = 1:niter
    % update W if requested
    if update_W
        W = W .* sqrt( ((Yp*H')+(Yn*W*(H*H'))) ./ ((Yn*H')+(Yp*W*(H*H'))) );
        if norm_w ~= 0
            W = normalize_W(W,norm_w);
        end
    end
    
    % update H if requested
    if update_H
        H = H .* sqrt( ((Yp*W)'+(W'*Yn*W*H)) ./ ((Yn*W)'+(W'*Yp'*W*H)) );
        if norm_h ~= 0
            H = normalize_W(H,norm_h);
        end
    end
    
    % compute squared error
    errs(t) = sum(sum((V-V*W*H).^2));
    
    % display error if asked
    if verb >= 3
        disp(['nmf_convex: iter=' num2str(t) ', err=' num2str(errs(t))]);
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
    disp(['nmf_convex: final_err=' num2str(errs(t))]);
end

% if we broke early, get rid of extra 0s in the errs vector
errs = errs(1:t);

% needed to conform to function signature required by nmf_alg
vout = {};
