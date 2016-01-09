function X = cell_to_matrix( cell_data ) 
	dd = [cell_data{:}];
	X = [dd{:}]';
