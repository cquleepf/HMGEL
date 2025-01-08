function M = CDLC(Ys, Yt_prob,GWs,GYs)
%   Cross Domain Local Consistancy,  minus distances between target samples
%   and coresponding source class centers; max distances between target
%   samples and source diff class center.

%   Inputs:
%%% Ys              : source label, ns * 1
%%% Yt_prob     : target posterior probability, nt * C
%%% GWs          : gb transform matrix, n * n_g
%%% GYs           : source gb labels, n_g * 1

%   Outputs:
%%% M             : cross domain matrix
%% 
if ~(nargin == 2 || nargin == 4)
    error(' Incorrect number of input arguments. Expected 2 or 4 arguments.');
end
nt = size(Yt_prob, 1);

%% construct one-hot matrix
[~,tt] = sort(Yt_prob,2);
tp2 = [full(sparse(1:nt,tt(:,1),1)), zeros(nt,max(Ys)-max(tt(:,1)))]';  % Yt^\top
tp3 = [full(sparse(1:nt,tt(:,2),1)), zeros(nt,max(Ys)-max(tt(:,2)))]';  % Zt^\top

if nargin==2
    Ys_onehot = full(sparse(1:length(Ys),Ys,1));
    map_matrix = Ys_onehot * diag(1./(eps+sum(Ys_onehot)));     % Ys(Ys^\topYs)^{-1}
    tp22 = [-map_matrix*tp2;eye(nt)];
    tp33 = [-map_matrix*tp3;eye(nt)];
end

if nargin ==4
    Ys_onehot = full(sparse(1:length(GYs),GYs,1));
    map_matrix = Ys_onehot * diag(1./(eps+sum(Ys_onehot)));     % Ys(Ys^\topYs)^{-1}
    tp22 = blkdiag(GWs,eye(nt))*[-map_matrix*tp2;eye(nt)];
    tp33 = blkdiag(GWs,eye(nt))*[-map_matrix*tp3;eye(nt)];
end

M = tp22*tp22' - tp33 * tp33';    % Eq(8) construct Mi matrix in Eq(9)

end

