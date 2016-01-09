function reduced_data = apply_pca(whole_data, U, n_dims)
	Ured = U(:,1:n_dims);
        for n_class=1:numel(whole_data)
        	for n_image=1:numel(whole_data{n_class})
                    reduced_data{n_class}{n_image} = Ured' * whole_data{n_class}{n_image};
	 	end
	end
end
