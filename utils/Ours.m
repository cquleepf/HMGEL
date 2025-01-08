function [acc,acc_ite,obj] = Method4(Xs,Ys,Xt,Yt,options)

    Xs=normr(Xs')';
    Xt=normr(Xt')';
    obj=[];
    acc=0;
    acc_ite=[];
    
    X=[Xs,Xt];
    X=L2Norm(X')';
    [m,ns]=size(Xs);
    nt=size(Xt,2);
    n=ns+nt;
    C=length(unique(Ys));
    dim=options.dim;                 
    aug=options.aug;                 
    lambda=options.lambda;           
    k=options.k;                     
    alpha1=options.alpha1;          
    alpha2=options.alpha2;           
    alpha3=options.alpha3;          
    tau=options.tau; 
    %% The neighborhood number should less than 'min(ns,nt)-1'
    if k>ns || k >nt
       fprintf('Replace k [%2d] as [%2d]\n',k,min(ns,nt)-1);
       k=min(ns,nt)-1; 
    end
    %% Decide whether initialize Ytpseudo
    Ytpseudo=[];
    if options.init==1
        fprintf('init pseudo-labels for target domain\n');
        if options.classify==1
            fprintf('Use KNN.\n');
            [Ytpseudo] = classifyKNN(Xs,Ys,Xt,1);
        else
           fprintf('Use SRM.\n');
           opt=options;
           [Ytpseudo,~,~] = SRM(Xs,Xt,Ys,opt);
        end
    else
         fprintf('init pseudo-labels for target domain as NULL\n');
    end
    %% init parameters for Ss
    clear opt;
    graphYs=hotmatrix(Ys,C,0);
    graphYs=graphYs*graphYs';
    opt.dist=my_dist(Xs');
    opt.semanticGraph=graphYs;
    opt.tau=tau;
    opt.k=k;
    [Ss_ori,rs]=AGE(opt); % ns * C
    
    %% init parameters for St
    if ~isempty(Ytpseudo)
        clear opt;
        opt.dist=my_dist(Xt');
        opt.tau=tau;
        opt.k=k;
        graphYtpseudo=hotmatrix(Ytpseudo,C)*hotmatrix(Ytpseudo,C)';
        opt.semanticGraph=graphYtpseudo;
        [St_ori,rt]=AGE(opt); % nt * C
    else
        St_ori=zeros(nt,nt);
        rt=0;
    end
    clear opt;
    M0=MMD0(Xs,Xt,C);   % the marginal distribution (modified)
    Ss=Ss_ori;
    St=St_ori;
    H=centeringMatrix(n);% the centering matrix
    right=X*H*X';% pre-calculation to save time

    %%%%%%%%%%%
    [GXs, GYs, GWs] = genEnvelope(Xs', Ys, options); % GXs : num * dim
    Yt_prob = [];
    %%%%%%%%%%%
    for i=1:options.T
        % calculate the whole S
        S=blkdiag(Ss,St);
        S=S+S';
        L = computeL_byW(S);
        if ~isempty(Ytpseudo)
            %%% Construct Raw sample MMD matrix
            N0 = MMDC(Ys,Ytpseudo,C);
            [~,d_m,d_c] = estimate_mu(Xs',Ys, Xt',Ytpseudo); % Estimate mu
            
            %%% Constuct GB MMD matrix
            [GXt, GYt, GWt] = genEnvelope(Xt', Ytpseudo, options);
            Mg = MMD0(GXs', GXt', C, GWs, GWt);
%             Ng = MMDC(GYs, GYt, C, GWs, GWt);
            [~,d_gm,~] = estimate_mu(GXs, GYs, GXt, GYt);
            
            d = d_m + d_c + d_gm;
            M = d_m/d * M0+ d_c/d*N0 + d_gm/d*Mg;
%             fprintf('contribute weights M_0:[%2f], M_c:[%2f], M_g:[%2f]\n',d_m/d,d_c/d,d_gm/d);
            
            %%% source intra-class variance
            Mi = zeros(n);
            Ms = IntraVar(Ys, 1, ns, nt);
            Mi = Mi + Ms;
            
            %%% source inter-class variance
            Mb = zeros(n);
            Mbs = InterVar(Ys, 1, ns, nt, C);
            Mb = Mb + Mbs;
            
            %%% Construct cross domain local learning and target LDA
            Me = 0;
            if ~isempty(Yt_prob)
                temp_Me = CDLC(Ys, Yt_prob, GWs, GYs); 
                Me = Me + temp_Me;
                Me = Me/(eps + norm(Me, 'fro'));
                
                Mt = IntraVar(Yt_prob, 0, ns, nt);
                Mi = Mi + Mt;
                
                Mbt = InterVar(Yt_prob, 0, ns, nt,C); 
                Mb = Mb + Mbt;
            end
            Mi = Mi/(eps + norm(Mi, 'fro'));
            Mb = Mb/(eps + norm(Mb, 'fro'));
        else
            % marginal distribution
            M=M0;
        end
        L=L./norm(L,'fro');
        M=M./norm(M,'fro');

        left=X*(M+alpha1*Me+alpha2*(Mi-Mb)/C+alpha3*L)*X'+lambda*eye(m);
        [A,~]=eigs(left,right,dim,'sm');
        AX=A'*X;
        AX=L2Norm(AX')';
        AXs=AX(:,1:ns);
        AXt=AX(:,ns+1:end);
        %% construct Yt_prob
        IDX = knnsearch(AXs', AXt', 'K', 5, 'distance', 'cosine');
        Yt_prob = zeros(size(IDX,1),length(unique(Ys)));
        for column = 1:size(Yt_prob,2)
            Yt_prob(:,column) = sum(Ys(IDX) == column,2);
        end
        Yt_prob = Yt_prob./5;
    
        if options.classify==1
            [Ytpseudo] = classifyKNN(AXs,Ys,AXt,1);
        else
           opt=options;
           [Ytpseudo,~,~] = SRM(AXs,AXt,Ys,options);
        end
        acc=getAcc(Ytpseudo,Yt);
        acc_ite(i)=acc;
         
        clear opt;
        opt.dist=my_dist(AXs');
        opt.semanticGraph=graphYs;
        opt.tau=tau;
        opt.k=k;
        opt.rr=rs; % 
        [Ss_iter,rs]=AGE(opt);
         
        clear opt;
        opt.dist=my_dist(AXt');
        opt.k=k;
         
        graphYtpseudo=hotmatrix(Ytpseudo,C)*hotmatrix(Ytpseudo,C)'; 
        opt.semanticGraph=graphYtpseudo;
        if options.init==0&& i==1
            % do nothing
        else
            opt.rr=rt; 
        end
        opt.tau=tau;
        [St_iter,rt]=AGE(opt);        
        Ss=(1-aug)*Ss+aug*Ss_iter;
         
        if options.init==0&& i==1
            St=St_iter;
        else
            St=(1-aug)*St+aug*St_iter;
        end
        %% print the results
        fprintf('Iteration:[%2d] acc:%.4f\n',i,acc*100);
        %% calculate the objective function
        obj(i)=trace(A'*left*A);
    end
end
function [L] = computeL_byW(W)
    n=size(W,1);
    Dw = diag(sparse(sqrt(1 ./ (sum(W)+eps) )));
    L = eye(n) - Dw * W * Dw;
end
function D = my_dist(fea_a,fea_b)
%% input:
%%% fea_a: n1*m
%%% fea_b: n2*m
    if nargin==1
        fea_b=0;
    end
    if nargin<=2
       [n1,n2]=size(fea_b);
       if n1==n2&&n1==1
           bSelfConnect=fea_b;
           fea_mean=mean(fea_a,1);
           fea_a=fea_a-repmat(fea_mean, [size(fea_a,1),1]);
           D=EuDist2(fea_a,fea_a,1);
           if bSelfConnect==0
                maxD=max(max(D));
                D=D+2*maxD*eye(size(D,1));
           end
           return ;
       end
    end
    fea_mean=mean([fea_a;fea_b],1);
    fea_a=fea_a-repmat(fea_mean, [size(fea_a,1),1]);
    fea_b=fea_b-repmat(fea_mean, [size(fea_b,1),1]);
    D=EuDist2(fea_a,fea_b,1);
end