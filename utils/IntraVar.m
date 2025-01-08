function M = IntraVar(Ys, is_Src, ns, nt)
%	Intra-Class Compactness is sigle domain, minus distances between
%	samples and corresponding class center
 	
%   Inputs:
%%% Ys       : label vector or label prob matrix. n*1 or n*C
%%% is_Src  : 1:Y is src label vector; 0:Y is tar prob matrix
%%% ns       : sample num of src
%%% nt       : sample num of tar

%   Outputs:
%%% M       : the intraVar matrix.  (ns+nt)*(ns+nt)

%% build Intra Class Variance 
if is_Src
    Y_prob = full(sparse(1:length(Ys),Ys,1));     % Ys one-hot matrix, ns-by-C
else
    Y_prob = Ys;
end

map_matrix = Y_prob * diag(1./(eps+sum(Y_prob)));     % num(tp) compute sample nums in each class; Ys(YsYs^\top)^{-1}
class_map = map_matrix*Y_prob';     % mapping class center to Y label space, Y *(Y  Y^\top)^{-1}* Y^\top   

if is_Src
    intra_var = [eye(ns); zeros(nt,ns)] - [eye(ns); zeros(nt,ns)] * class_map;   % [I-Ys(Ys^\top*Ys+)^{-1}Y_s^\top; 0]
    M = intra_var*intra_var';
else
    intra_var = [zeros(ns,nt);eye(nt)] - [zeros(ns,nt);eye(nt)]*class_map; % [0;I-Yt(Yt^\top*Yt+)^{-1}Y_t^\top]
    M = diag([zeros(ns,1);ones(nt,1)])*(intra_var*intra_var');
end

end
