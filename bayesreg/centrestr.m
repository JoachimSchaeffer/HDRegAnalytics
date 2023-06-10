function sc = centrestr(s, w)
%CENTRESTR pad a string with spaces to centre it string 
%   s = repchar(...) pads a string with spaces to centre it
%
%   The input arguments are:
%       s      - string to centre
%       w      - number of characters in which to centre the string
%
%   Return values:
%       s     - centred string
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2017

l = length(s);
sc = sprintf('%s%s%s', repchar(' ', floor(w/2) - floor(l/2)), s, repchar(' ', ceil(w/2) - ceil(l/2)));

return