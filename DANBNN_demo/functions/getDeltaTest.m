function [ delta ] = getDeltaTest( S, te, testSample, c )
%GETDELTATEST Summary of this function goes here
%   Detailed explanation goes here
    ll=find(S.label==c);%images of class c
    ts=cell(numel(ll),1);
    [ts{:}]=deal(S.feat{ll});
    feat_class=cell2mat(ts');
    delta = getPatchDistance( feat_class, te{testSample});
end

