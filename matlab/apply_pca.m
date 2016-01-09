function reduced_data = apply_pca(whole_data, U, n_dims, xmean, xstd)
	Ured = U(:,1:n_dims);
        for n_class=1:numel(whole_data)
        	for n_image=1:numel(whole_data{n_class})
		    tmp = whole_data{n_class}{n_image}';
		    n = size(tmp, 1)
		    tmp = (tmp - repmat(xmean,[n 1])) ./ repmat(xstd,[n 1]); 
                    reduced_data{n_class}{n_image} = tmp*Ured;
	 	end
	end
end
