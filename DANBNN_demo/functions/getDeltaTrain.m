function [ delta ] = getDeltaTrain( S, testSample, c )
%GETDELTATRAIN Summary of this function goes here
%   Detailed explanation goes here
    ll=find(S.label==c);%images of class c
    feat_te=S.feat{testSample};
    if (ll~=testSample) %if different class
        %do nothing
    else %if the class is the same of the sample, we must remove the sample from the support patches (else distance is always 0)
        idx=find(ll==testSample);
        ll(idx)=[];
    end
    ts=cell(numel(ll),1);
    [ts{:}]=deal(S.feat{ll});
    feat_class=cell2mat(ts');
    delta = getPatchDistance( feat_class, feat_te);
end

