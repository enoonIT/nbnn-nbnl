
% This code is part of the supplementary material to the ICCV 2013 paper
% "Frustratingly Easy NBNN Domain Adaptation", T. Tommasi, B. Caputo. 
%
% Copyright (C) 2013, Tatiana Tommasi
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
%
% Please cite:
% @inproceedings{TommasiICCV2013,
% author = {Tatiana Tommasi, Barbara Caputo},
% title = {Frustratingly Easy NBNN Domain Adaptation},
% booktitle = {ICCV},
% year = {2013}
% }
%
% Contact the author: ttommasi [at] esat.kuleuven.be
%


function M=fn_create_metric(DD,dec,label,gamma,Ns)

lambda=0.5;

N=numel(label);
Nt=N-Ns;

uy=unique(label);

for c=uy 
    s1=0;
    s2=0;
    ind1=find(label(1:Ns)==c);
    ind2=find(label(Ns+1:end)==c)+Ns;
    for i=1:numel(ind1)
        s1=s1+DD{ind1(i),label(ind1(i))};
    end
    for i=1:numel(ind2)
        s2=s2+DD{ind2(i),label(ind2(i))};
    end
    
    G{c}=(1-lambda)*(gamma*s1+(1-gamma)*s2);
    M{c}=eye(size(DD{1,1})); 
   clear ind1 ind2 s1 s2
end

fprintf(1,'Metric optimization ');
out={'|' '/' '-' '\'};

for t=1:200  % limited number of rounds for the demo
    if mod(t,4)==0
        fprintf(1,'\b%s',out{4});
    else
        fprintf(1,'\b%s',out{mod(t,4)});
    end

    for i=1:N
        p(i)=trace(dec{i,label(i)}*M{label(i)}*dec{i,label(i)}');
        for c=uy
            n(c)=trace(dec{i,c}*M{c}*dec{i,c}');
            xi(i,c)=1-n(c)+p(i);
        end
        clear n
        xi(i,label(i))=-Inf;
        idx{t,i}=find(xi(i,:)>=0); %indicates which negative class is active for a fixed (i,p)
        
        if t>1
            comp=ismember(idx{t,i}(:), idx{t-1,i}(:));
            new{i}=idx{t,i}(~comp); %new active triplets;
            comp=ismember(idx{t-1,i}(:),idx{t,i}(:));
            old{i}=idx{t-1,i}(~comp); %disactivated triplets;
        else
            new{i}=idx{t,i}(:);
            old{i}=[];
        end
    end
    str1=sum(p(1:Ns));
    str2=sum(p(Ns+1:end));
    clear p
    
    for c=uy
        
        % part from source %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        s1=zeros(size(DD{1,1}));
        s2=zeros(size(DD{1,1}));
        s3=zeros(size(DD{1,1}));
        s4=zeros(size(DD{1,1}));

        ind_p=find(label(1:Ns)==c);
        ind_n=find(label(1:Ns)~=c);
        for i=1:numel(ind_p)
            s1=s1+numel(new{ind_p(i)})*DD{ind_p(i),c};
            s3=s3+numel(old{ind_p(i)})*DD{ind_p(i),c};
        end
        for i=1:numel(ind_n)
            test1=find(new{ind_n(i)}==c);
            if test1
                s2=s2+DD{ind_n(i),c};
            end
            test2=find(old{ind_n(i)}==c);
            if test2
                s4=s4+DD{ind_n(i),c};
            end
        end
        part1= gamma*(lambda*(s1-s2)-lambda*(s3-s4));
        clear ind_p ind_n s1 s2 s3 s4 test1 test2
        
        %part from target %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        s1=zeros(size(DD{1,1}));
        s2=zeros(size(DD{1,1}));
        s3=zeros(size(DD{1,1}));
        s4=zeros(size(DD{1,1}));

        ind_p=find(label(Ns+1:end)==c)+Ns;
        ind_n=find(label(Ns+1:end)~=c)+Ns;
        for i=1:numel(ind_p)
            s1=s1+numel(new{ind_p(i)})*DD{ind_p(i),c};
            s3=s3+numel(old{ind_p(i)})*DD{ind_p(i),c};
        end
        for i=1:numel(ind_n)
            test1=find(new{ind_n(i)}==c);
            if test1
                s2=s2+DD{ind_n(i),c};
            end
            test2=find(old{ind_n(i)}==c);
            if test2
                s4=s4+DD{ind_n(i),c};
            end
        end
        part2= (1-gamma)*(lambda*(s1-s2)-lambda*(s3-s4));
        clear ind_p ind_n s1 s2 s3 s4 test1 test2
        
        G{c}=G{c}+part1+part2; 
        part1=0;
        part2=0;
        M{c}=M{c}-G{c}/(sqrt(t)*N);
        
        % project the matrix to keep it positive semidefinite
        [vec,val]=eig(M{c});
        M{c}=vec*(max(val,0))*inv(vec);
    end
    
    xi_pos=max(xi,0);
    xi_pos1=sum(sum(xi_pos(1:Ns,:)));
    xi_pos2=sum(sum(xi_pos(Ns+1:end,:)));
    
    obj(t)=gamma*((1-lambda)*str1+lambda*xi_pos1)+(1-gamma)*((1-lambda)*str2+lambda*xi_pos2);

    st=1:99:200;
    if ismember(t,st)
        fprintf(1,'\b... '); 
        %fprintf('[ %d %f ]\n', t, obj(t));
    end

    clear xi xi_pos
    if t>1 && obj(t-1)-obj(t)>0 && obj(t-1)-obj(t)<0.1
        break;
    end
end
fprintf(1,'\b\n'); 

end
