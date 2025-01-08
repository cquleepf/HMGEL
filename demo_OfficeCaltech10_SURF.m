clc; clear all;
addpath(genpath('./utils/'));
rng(2, 'twister' );  

srcStr = {'caltech','caltech','caltech','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
tgtStr = {'amazon','webcam','dslr','caltech','webcam','dslr','caltech','amazon','dslr','caltech','amazon','webcam'};
finalResult=[];

%% initialize the Ytpseudo by a basic classifier or not 
options.init=1; % if `options.init==1`, then Ytpseudo is assigned before training
options.classify=2; % `options.classify==1`, use KNN
                    % Otherwise, use SRM
options.gamma=1; % The parameter of SRM
options.Kernel=2; % The parameter of SRM
options.mu=0.1; % The parameter of SRM
options.k=32; % neighborhood number  
options.tau=1e-3; 
%% The hyper-parameters of AGE-CS
options.T=10; % iteration
options.purity=0.8;
options.dim=50; % dimension  
options.alpha1=0.01;
options.alpha2=1;
options.alpha3=5;  
options.lambda=0.1; % the weight of regularization for projection matrix

%% Run the experiments
for i = 1:12
    src = char(srcStr{i});
    tgt = char(tgtStr{i});
    fprintf('%d: %s_vs_%s\n',i,src,tgt);
    load(['./data/OfficeCaltech10_SURF/' src '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
    Xs =zscore(fts,1);
    Ys = labels;
    load(['./data/OfficeCaltech10_SURF/' tgt '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
    Xt = zscore(fts,1);
    Yt = labels;
    Xs=Xs';
    Xt=Xt';
    [~,result,~] = Ours(Xs,Ys,Xt,Yt,options);
    finalResult=[finalResult;result];
end

result_aver = mean(finalResult(:,end));
list_acc = [finalResult(:,end);result_aver]*100
options.purity

fid = fopen('./results/exp1_OfficeCaltech_SURF.csv', 'a');
fprintf(fid,'%0.4f,',list_acc);
fprintf(fid, '\n');
fclose(fid);
