
clc
clear 
close all
run('VLFEATROOT/vlfeat-0.9.21-bin/vlfeat-0.9.21/toolbox/vl_setup')

data_folder = 'C:/Users/johny/Desktop/Visual Analysis/Projects/2 - Recognition with a Vocabulary Tree/data2';

server_folder =  fullfile(data_folder,'server');
client_folder =  fullfile(data_folder,'client');


imgs_server = fullfile(server_folder, '*.jpg');
imgs_server = dir(imgs_server);
s=fieldnames(imgs_server);
imgs_server = rmfield(imgs_server,s(2:end));
imgs_server = struct2cell(imgs_server);

imgs_client = fullfile(client_folder, '*.jpg');
imgs_client = dir(imgs_client);
s=fieldnames(imgs_client);
imgs_client = rmfield(imgs_client,s(2:end));
imgs_client = struct2cell(imgs_client);

if ~exist('data2/server_features', 'dir')
   mkdir('data2/server_features')
end

if ~exist('data2/client_features', 'dir')
   mkdir('data2/client_features')
end

%% Sift object
PeakThresh = 0.025;
EdgeThresh = 5;

PeakThresh = 0.005;
EdgeThresh = 3;

Sift_obj = Sift_det(PeakThresh, EdgeThresh);



%%% extracting server features
server_dict = containers.Map('KeyType', 'char', 'ValueType', 'any');
server_features_count = zeros(1,50);

for index = 1:length(imgs_server)
    index
    object_id = split(imgs_server{index},"_");
    object_id = object_id{1};
    object_index = split(object_id,'obj');
    object_index = str2double(object_index{2});

    current_img_path = fullfile(server_folder,imgs_server{index});
    %%% Extracting features server
    I = imread(current_img_path);
    I = rgb2gray(I);
    I=im2single(I);
    [Psift, Fsift] = Sift_obj.extract_points(I);
    
    
    if isKey(server_dict,object_id)
        server_dict(object_id) = cat(1,server_dict(object_id),Fsift);
    else
        server_dict(object_id) = Fsift;
    end
    server_features_count(object_index) = server_features_count(object_index) + size(Fsift,1);
%     close all
end

%%% saving extractings features
for current_key = keys(server_dict)
    current_key = current_key{1};
    save_path = fullfile('data2/server_features',current_key);
    current_feat = server_dict(current_key);
    save(save_path + ".mat",'current_feat');
end

%%% extracting client features
client_dict = containers.Map('KeyType', 'char', 'ValueType', 'any');
client_features_count = zeros(1,50);

for index = 1:length(imgs_client)
    index
    object_id = split(imgs_client{index},'_');
    object_id = object_id{1};
    object_index = split(object_id,'obj');
    object_index = str2double(object_index{2});

    current_img_path = fullfile(client_folder,imgs_client{index});
    %%% Extracting features server
    I = imread(current_img_path);
    I = rgb2gray(I);
    I=im2single(I);
    [Psift, Fsift] = Sift_obj.extract_points(I);
    
    
    if isKey(client_dict,object_id)
        client_dict(object_id) = cat(1,client_dict(object_id),Fsift);
    else
        client_dict(object_id) = Fsift;
    end
    client_features_count(object_index) = client_features_count(object_index) + size(Fsift,1);
end

%%% saving extractings features
for current_key = keys(client_dict)
    current_key = current_key{1};
    save_path = fullfile('data2/client_features',current_key);
    current_feat = client_dict(current_key);
    save(save_path + ".mat",'current_feat');
end









