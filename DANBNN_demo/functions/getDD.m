function [ delta2 ] = getDD( S, testSample, c )
%GETDD Summary of this function goes here
%   Detailed explanation goes here
    delta = getDeltaTrain(S,testSample,c);
    delta2 = delta'*delta;

end

