function [varargout] = parse_opt(args, varargin)
% function [varargout] = parse_opt(args, varargin)
%
% Process name-value argument pairs which can be passed in arbitrary
% order.  This function is very much like Mark A. Paskin's
% 'process_options' function (from which it borrows heavily), but aims 
% to be a bit faster and therefore is simpler and does less error checking.
%
% Example: Suppose we pass varargin = {'a', 23, 'c', 'hi'} into parse_opt:
%
%    [a,b,c] = parse_opt(varargin, 'a', 1, 'b', 2, 'c', 'test');
%
% This would result in a=23, b=2, and c='hi' as we gave values for variables 
% 'a' and 'c', but not 'b' which got the default value of 2.
% 
% 2010-01-14 Graham Grindlay (grindlay@ee.columbia.edu)

% Copyright (C) 2008-2028 Graham Grindlay (grindlay@ee.columbia.edu)
% Based on process_options.m Copyright (C) 2002 Mark A. Paskin
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

ni = length(varargin);

% these are just about the only things that we'll check for
if nargout*2 ~= ni
    error('parse_opt requires a name-value input pair for each output');
end
if mod(ni,2) ~= 0
     error('parse_opt requires name-value input pairs');
end

no = ni/2;
varargout = cell(1,no);
for i = 1:2:ni
    ndx = find(strcmpi(varargin{i}, args));
    if isempty(ndx)
        varargout{(i+1)/2} = varargin{i+1};
    else
        varargout{(i+1)/2} = args{ndx+1};
    end
end
