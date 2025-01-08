function [N] = MMDC(Ys, Yt, C, Ws, Wt)
% TODO : Unified MMD function
% Marginal MMD matrix for Ball and Centers
% Inputs:
%%% Xs  : Source domain feature matrix, dim * ns
%%% Xt  : Target domain feature matrix, dim * nt
%%% Ws  : Source domain transform matrix, ns * new_ns
%%% Wt  : Source domain transform matrix, nt * new_nt
%%% C   : the number of labels
%%% new_X = X * W
N=0;
ns = length(Ys);
nt = length(Yt);

if nargin == 3 
    Ws = eye(ns);
    Wt = eye(nt);
end

for c = 1:C
    e = zeros(ns+nt,1);
    e(Ys==c) = 1/length(find(Ys==c));
    e(ns+find(Yt==c)) = -1/length(find(Yt==c));
    e = blkdiag(Ws,Wt) * e;
    e(isinf(e)) = 0; 
    N = N + e * e';
end

end

