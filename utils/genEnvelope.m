function [GX,GY,W] = genEnvelope(X, Y, options)
% Inputs:
%%% X       : raw data, n*dim
%%% Y       : raw data labels, n*1
%%% options : only the purity of ball, options.purity, (05--1.0)

% Outputs:
%%% GX      : centroids of the Envelope, n_g * dim
%%% GY      : labels of the ball, is the maximum number of samples in the ball , n_g * 1
%%% W       : transform maxtrix,  n * n_g
%%% Usage : X * W = Xg  === dim-by-n * n-by-ng = dim-by-ng

data = [double(Y), X];

[~,d_idx,~] = unique(data(:,2:end),'row','stable');
data = data(d_idx,:);
% init
center_init = data(randi([1,size(data,1)]), :); % select a center randomly
dist_init = sqrt(sum((data(:,2:end) - center_init(2:end)).^2, 2));  % calculate distancs between all samples and center

% create an empty struct, including center, all samples, all distances
gb_dict = struct('center',{},'samples',{},'dists',{});
gb_dict(1).center = center_init;
gb_dict(1).samples = data;
gb_dict(1).dists = dist_init; 

gb_dict_new = struct('center',{},'samples',{},'dists',{});
while true
    % Copy a temporary list, and then traverse the value
    if isempty(gb_dict_new)
        gb_dict_temp = gb_dict; % initial assignment
    else
        gb_dict_temp = gb_dict_new; % Subsequent traversal assignment
    end
    gb_dict_new = gb_dict_temp;
    % record the number before a division
    ball_number_1 = length(gb_dict_temp);
    
    for i=1:length(gb_dict_temp)
        single_ball = gb_dict_temp(i);
        % compute current gb purity
        [~,p] = get_label_and_purity(single_ball.samples);
        % Determine whether the purity meets the requirements, if not,
        % continue to divide
        if p < options.purity && length(single_ball.samples) ~= 1
            gb_dict_split = splits_ball(single_ball);
            %% update gb
            % Iterate each element in split_dict
            for j = 1:length(gb_dict_split)
                % Determine whether there is a match center
                match_found = false;
                for k = 1:length(gb_dict_new)
                    if isequal(gb_dict_split(j).center, gb_dict_new(k).center)
                        % If there is a matching center, assign the values of the samples and dits fields to the corresponding new_dict.
                        gb_dict_new(k).samples = gb_dict_split(j).samples;
                        gb_dict_new(k).dists = gb_dict_split(j).dists;
                        match_found = true;
                        break;
                    end
                end
                
                % If there is no matching center, the entire element is added to new_dict.
                if ~match_found
                    gb_dict_new(end+1) = gb_dict_split(j);
                end
            end
            
        else
            %% update gb
            for j = 1:length(gb_dict_split)
                %  Determine whether there is a match center
                match_found = false;
                for k = 1:length(gb_dict_new)
                    if isequal(gb_dict_split(j).center, gb_dict_new(k).center)
                        % If there is a matching center, assign the values of the samples and dits fields to the corresponding new_dict.
                        gb_dict_new(k).samples = gb_dict_split(j).samples;
                        gb_dict_new(k).dists = gb_dict_split(j).dists;
                        match_found = true;
                        break;
                    end
                end
                
                %  If there is no matching center, the entire element is added to new_dict.
                if ~match_found
                    gb_dict_new(end+1) = gb_dict_split(j);
                end
            end
            
            continue;
        end
    end
    ball_number_2 = length(gb_dict_new);
    if ball_number_1 == ball_number_2
        break
    end
end
init_c = [];
for i=1:length(gb_dict_new)
    init_c=[init_c; gb_dict_new(i).center];
end
[Idx,~,~,~]=kmeans(gb_dict.samples(:,2:end),length(gb_dict_new),'Start',init_c(:,2:end),'rep',1);
temp_struct = struct('samples',{},'ball_label',{},'idx',{},'center',{});

datas=[data, (1:size(data,1))']; % first column label; the end column is IDX
uniqueLabels = unique(Y);
BallArray = cell(1, length(uniqueLabels)); % init a cell array��each element is used to store from label 1 to C gb idx
GX = []; GY = [];W = zeros(size(X,1),length(gb_dict_new));
for i=1:length(gb_dict_new)
    temp_struct(end+1).samples = datas(Idx==i,:);
    [currentLabel, ~] = get_label_and_purity( datas(Idx==i,1:end-1) );  
    temp_struct(end).ball_label = currentLabel;
    temp_struct(end).idx = datas(Idx==i,end);
    temp_struct(end).center = mean( data(Idx==i,2:end), 1 );
    GX = [GX; temp_struct(end).center];
    GY = [GY; currentLabel];
    %%% *************** find data idx in the X *************
    orign_idx = d_idx(Idx==i);
    W(orign_idx,i)=1 / length(find(Idx==i));
    W(isinf(W)) = 0;
    col = find(uniqueLabels == currentLabel);
    % merge dix
    if isempty(BallArray{col})
        BallArray{col} = temp_struct(i).idx;
    else
        BallArray{col} = [BallArray{col}; temp_struct(i).idx];
    end
end
% gb_plot(temp_struct);
end


function [new_gb] = splits_ball(gb_dict)

samples_all = gb_dict(1).samples;
center_old = gb_dict(1).center;
dists_old = gb_dict(1).dists;

uniq_label = unique(samples_all(:, 1), 'rows');
% When the input has only one type of data, select a point different from
% the original center
if length(uniq_label) > 1
    gb_class = length(uniq_label);
else
    gb_class = 2;
end

centers_new = [];
for i=1:gb_class-1
    if length(uniq_label) < 2        
        diff_samples = samples_all(dists_old ~= 0, :);  % Remove the old center
        ran = randi([1, size(diff_samples, 1)]);
        center_temp = diff_samples(ran, :);  % Take a new center
        centers_new = [centers_new; center_temp];
    else        
        if ismember(center_old(1), uniq_label)
            uniq_label(uniq_label == center_old(1)) = [];
        end
        % Extract heterogeneous data
        diff_samples = samples_all(samples_all(:, 1) == uniq_label(i), :);
        %% random center of heterogeneity, 
        ran = randi([1, size(diff_samples, 1)]);
        center_temp = diff_samples(ran, :);
        centers_new = [centers_new; center_temp];
    end
end

centers_all = [center_old; centers_new];

dists_new = []; 
for i=1:size(centers_new, 1)
    dists_new=[dists_new, sqrt(sum((samples_all(:,2:end) - centers_new(i,2:end)).^2, 2))];
end

dists_all = [dists_old, dists_new];

% 0:old center; 1,2...: new centers
[~, index_min] = min(dists_all,[],2);

new_gb = struct('center',{},'samples',{},'dists',{});
for i=1:size(centers_all,1)
    new_gb(end+1).center = centers_all(i,:);
    new_gb(end).samples = samples_all(index_min==i,:);
    new_gb(end).dists = dists_all(index_min==i,i);
end
end


function [label, purity] = get_label_and_purity(samples)
% Calculate the label and purity. Calculate the data with the highest
% proportion in the cluster, where the input is all samples in the cluster,
% and the last column is the label Calculate the number of data categories
[uniq_label, ~, counts] = unique(samples(:, 1));
counts = histcounts(counts);
if length(uniq_label) == 1
    purity = 1.0;
    label = uniq_label;
else
    [max_count, max_idx] = max(counts);
    purity = max_count / sum(counts);
    label = uniq_label(max_idx);
end

end


function gb_plot(gb_dict)
color = containers.Map([-1, 1, 0], {'k', 'r', 'b'});
axis([-1.2, 1.2, -1, 1]);
for i=1:length(gb_dict)
    samples = gb_dict(i).samples;
    [label, ~] = get_label_and_purity(gb_dict(i).samples);
    c = mean(samples(:,2:end),1);
    r = mean( sqrt(sum((samples(:,2:end) - c).^2, 2)) );
    data1 = samples(samples(:,1) == -1,:);
    data2 = samples(samples(:,1) == 1,:);
    
    scatter(data1(:,2),data1(:,3),[],'k','filled');
    hold on
    scatter(data2(:,2),data2(:,3),[],'r','filled');
    hold on
    
    theta = 0:pi/20:2*pi; 
    x = c(1)+r*cos(theta);
    y = c(2)+r*sin(theta);
    plot(x,y,'Color', color(label), 'LineWidth', 0.8);
    xlim([-1.2 1.2])
    ylim([-1 1])
    hold on
end

end
