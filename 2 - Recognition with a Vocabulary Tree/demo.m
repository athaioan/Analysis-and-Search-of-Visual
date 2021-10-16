clc
clear 
close all

data_folder = 'C:/Users/johny/Desktop/Visual Analysis/Projects/2 - Recognition with a Vocabulary Tree/data2';

server_folder =  fullfile(data_folder,'server_features');
client_folder =  fullfile(data_folder,'client_features');


features_server = fullfile(server_folder, '*.mat');
features_server = dir(features_server);
s=fieldnames(features_server);
features_server = rmfield(features_server,s(2:end));
features_server = struct2cell(features_server);

features_client = fullfile(client_folder, '*.mat');
features_client = dir(features_client);
s=fieldnames(features_client);
features_client = rmfield(features_client,s(2:end));
features_client = struct2cell(features_client);

%%% extracting server features
server_dict = containers.Map('KeyType', 'char', 'ValueType', 'any');
client_dict = containers.Map('KeyType', 'char', 'ValueType', 'any');

server_array = [];
client_array = [];

server_object = [];
client_object = [];


%%% loading server and client extracted featurs
for index = 1:length(features_server)
    
    current_key = features_server{index};
    current_key_int = split(current_key,'.');
    current_key_int = current_key_int(1);
    current_key_int = split(current_key_int,'obj');
    
    current_key_int = str2num(cell2mat(current_key_int(2)));

    
    load_path_server = fullfile('data2/server_features',current_key);
    load_path_client = fullfile('data2/client_features',current_key);

    current_feat = cell2mat(struct2cell(load(load_path_server)));
    server_dict(current_key) = current_feat;
    server_array(end+1:end+size(current_feat,1),:) = current_feat;
    server_object(end+1:end+size(current_feat,1),:) = ones(size(current_feat,1),1)*current_key_int;
    
    current_feat = cell2mat(struct2cell(load(load_path_client)));
    client_dict(current_key) = current_feat;
    client_array(end+1:end+size(current_feat,1),:) = current_feat;
    client_object(end+1:end+size(current_feat,1),:) = ones(size(current_feat,1),1)*current_key_int;

end


leaves_only = false;
client_percentage = 0.5;

b=5;
depth=7;
% b=4;
% depth=5;

rng(1); 

server_data.features = server_array;
server_data.objects = server_object;
server_data.n_objects = 50;


idx = randperm(size(client_array,1));
target_num_features = round(client_percentage * size(client_array,1));
idx = idx(1:target_num_features);

client_array = client_array(idx,:);
client_object = client_object(idx);


client_data.features = client_array;
client_data.objects = client_object;
client_data.n_objects = 50;

rng(1); 
%%% constucting tree
tree = hi_kmeans(server_data, b, depth);



%%% constructing the frequency matrix FA
FA = extract_FA(tree,server_data.n_objects,b,depth);
num_feats = FA(1,:);

% FA_query = extract_FA_query(client_data,tree,b,depth);
FA_query = extract_FA_query(client_data,tree,b,depth);
num_feats_query = FA_query(1,:);

if leaves_only 
    FA(end-b^depth:end,:);
    FA_query(end-b^depth:end,:);
end


%%% constructing tf-idf for database
%% idf 
idf = log2(server_data.n_objects./sum((FA>0),2));

%% tf 
tf = FA./num_feats;
tf_idf_database =  tf .* idf;


%% there might be cases that 0 clusters reached a node
irrelevant_weights = idf==inf;
tf_idf_database(irrelevant_weights,:) = [];

%% applying tree on query data
tf_query = FA_query./num_feats_query;
tf_idf_client =  tf_query .* idf;
tf_idf_client(irrelevant_weights,:) = [];


%%% l1-norm 
L1_dist = pdist2(tf_idf_client',tf_idf_database','cityblock');
L1_dist = pdist2(tf_idf_client',tf_idf_database','spearman');
% L1_dist = pdist2(tf_idf_client',tf_idf_database','cosine');

top_n = 1;

top_retrievals = zeros(client_data.n_objects,top_n);

for index_top=1:top_n
  [~,min_indices] = min(L1_dist,[],2);
  % remove for the next iteration the last smallest value:
  pair_indices = sub2ind(size(L1_dist),(1:client_data.n_objects)',min_indices);
  L1_dist(pair_indices) = inf;
  top_retrievals(:,index_top) = min_indices;
end

% evaluate recal rate
ground_truth = (1:client_data.n_objects)';
recall_errors = abs(top_retrievals-ground_truth);

recall_rate = sum(min(recall_errors,[],2)==0)/client_data.n_objects

x=1;

function tree = hi_kmeans(data, b, depth)
%     n_nodes = b ^ (depth+1) - 1;                                    
    size(data.features,1);

    node=struct();
    max_depth = depth;
    tree = split_level(node,data,b,depth,max_depth);
        
end

function node = split_level(node,data,b,depth,max_depth)

    node.depth = max_depth-depth;
    if depth > 0
        
        kmeans_b = min(b,size(data.features,1));
       
        [idx, C] = kmeans(data.features,kmeans_b,'MaxIter',200);

        node.centroids = C;
        
        depth = depth - 1;
        
        node.children={};
        
        

        for current_child=1:kmeans_b
            current_idx = (idx == current_child);
            current_data.features = data.features(current_idx,:);
            size(current_data.features,1);
            current_data.objects = data.objects(current_idx,:);
            current_data.n_objects = data.n_objects;


            child_node = struct();           
            child_node = split_level(child_node,current_data,b,depth,max_depth);
            
            node.children{end+1} = child_node;
        end
    else
        node.objects_freq = histcounts(data.objects,1:data.n_objects+1);
    end

    
end

function FA = extract_FA(tree,n_objects,b,d)
    
    N = (1 - b^(d+1))/(1-b); % number of tree nodes
    
    FA = zeros(N,n_objects);
    
    path=[];
 
    FA = fill_FA(tree,FA,b,path);
      
end

function FA = fill_FA(tree,FA,b,path)

   if isfield(tree,'children')
       for index = 1:length(tree.children)
            path_current = path;
            path_current(end+1) = index;
            FA = fill_FA(tree.children{index},FA,b,path_current);
       end
       
       
   else
       objects_freq = tree.objects_freq;
       d = tree.depth;
       
       indices = [];
       while length(path>1)

           N_ = (1 - b^(d))/(1-b);

           index = path(end);
           path_temp = path(1:end-1)-1;
           e = d-1:-1:1;
           e = b.^e;

           index = N_ + sum(e.*path_temp) + index;
           indices(end+1) = index;
           d = d-1;
           path(end)=[];
       end
       indices(end+1) = 1;
%        indices
       FA(indices,:) = FA(indices,:)+ objects_freq;
   end


end


function FA = extract_FA_query(data,tree,b,d)
    
    N = (1 - b^(d+1))/(1-b); % number of tree nodes
    
    n_objects = data.n_objects;
    FA = zeros(N,n_objects);
    
    path=[];

    FA = fill_FA_query(data,tree,FA,b,path);
      
end


function FA = fill_FA_query(data,tree,FA,b,path)

   if isfield(tree,'children')
       
       d = tree.depth;
       centroids = tree.centroids;

       
       if length(path)>0
           N_ = (1 - b^(d+1))/(1-b);
           current_index = path(end);
           e = d:-1:1;
           e = b.^e;
           N_ = N_ + sum(e.*(path-1));
       else
          N_ = 1;
          objects_freq = histcounts(data.objects,1:data.n_objects+1);
          FA(N_,:) = objects_freq;
       end
       
       D = pdist2(data.features,centroids);
       [~,idx] = min(D,[],2);
       
       for child_index = 1:length(tree.children)
            
           
           current_data.features = data.features(idx==child_index,:);
           current_data.objects = data.objects(idx==child_index);
           current_data.n_objects = data.n_objects;
           objects_freq = histcounts(current_data.objects,1:current_data.n_objects+1);

           FA(child_index+N_,:) = objects_freq;
           path_current = path;
           path_current(end+1) = child_index;
           FA = fill_FA_query(current_data,tree.children{child_index},FA,b,path_current);
       end
       
       
   else
        return
   end


end
