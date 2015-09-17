function myAddPath( folder )
%MYADDPATH Summary of this function goes here
%   Detailed explanation goes here
    if(not(isdeployed))
        addpath(folder);
    end

end

