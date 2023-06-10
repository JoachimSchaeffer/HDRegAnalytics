function s = repchar(code, n)
%REPCHAR print string code n times to a string.
%   s = repchar(...) prints the string code, n times, to a string.
%
%   The input arguments are:
%       code   - string to print n times
%       n      - [1 x 1] how many times to repeat 
%
%   Return values:
%       s     - string containing code repeated n times
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

s = sprintf('%c', ones(1,n)*code);

end