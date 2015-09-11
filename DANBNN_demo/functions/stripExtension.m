function [ basename ] = stripExtension( filename )
%STRIPEXTENSION Removes the file extensions and returns the base name
%   Detailed explanation goes here
    pattern = '\.[^.]+$';
    replacement = '';
    basename = regexprep(filename,pattern,replacement);
end

