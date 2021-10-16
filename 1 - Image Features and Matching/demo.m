clc
clear 
close all
run('VLFEATROOT/vlfeat-0.9.21-bin/vlfeat-0.9.21/toolbox/vl_setup')


%% Part 2
I = imread('data1/obj1_5.JPG');
I = rgb2gray(I);
I=im2single(I);

%% Surf repetability
%% Extracting the key points
MetricThreshold = 6000;

rot_step = 15;
scale_step = 1.2;

Surf_obj= Surf_det(MetricThreshold, rot_step, scale_step);
Psurf = Surf_obj.extract_points(I);
surf_repeatability_angle = Surf_obj.compute_repeatability_angle(I, Psurf);
figure()
plot(0:Surf_obj.rot_step:360 ,surf_repeatability_angle) 
title('Surf repeatability vs Rotation')
xlabel('Angle in degrees') 
ylabel('Repeatability') 

surf_repeatability_scale = Surf_obj.compute_repeatability_scale(I, Psurf);
figure()
plot(power(Surf_obj.scale_step,0:8) ,surf_repeatability_scale) 
title('Surf repeatability vs Scale')
xlabel('Scale') 
ylabel('Repeatability') 



%% Sift repetability
PeakThresh = 0.047;
EdgeThresh = 5;
% 
rot_step = 15;
scale_step = 1.2;

Sift_obj= Sift_det(PeakThresh, EdgeThresh, rot_step, scale_step);
Psift = Sift_obj.extract_points(I);
% % 
sift_repeatability_angle = Sift_obj.compute_repeatability_angle(I, Psift);
figure()
plot(0:Sift_obj.rot_step:360 ,sift_repeatability_angle) 
title('Sift repeatability vs Rotation')
xlabel('Angle in degrees') 
ylabel('Repeatability') 


sift_repeatability_scale = Sift_obj.compute_repeatability_scale(I, Psift);
figure()
plot(power(Sift_obj.scale_step,0:8) ,sift_repeatability_scale) 
title('Sift repeatability vs Scale')
xlabel('Scale') 
ylabel('Repeatability') 


% % % figure()
% % % hold on
% % % plot(0:Surf_obj.rot_step:360 ,surf_repeatability_angle) 
% % % plot(0:Sift_obj.rot_step:360 ,sift_repeatability_angle) 
% % % title('Repeatability vs Rotation')
% % % xlabel('Angle in degrees') 
% % % ylabel('Repeatability') 
% % % legend('SURF)','SIFT')


%% Part 3 - Sift
PeakThresh = 0.047;
EdgeThresh = 5;

rot_step = 15;
scale_step = 1.2;

Sift_obj= Sift_det(PeakThresh, EdgeThresh, rot_step, scale_step);

I_query = imread('data1/obj1_5.JPG');
I_query = rgb2gray(I_query);
I_query=im2single(I_query);

I_data = imread('data1/obj1_t1.JPG');
I_data = rgb2gray(I_data);
I_data=im2single(I_data);


[Psift_query,features_query] = Sift_obj.extract_points(I_query);

[Psift_data,features_data] = Sift_obj.extract_points(I_data);

% 
% % [id_query, id_data] = getMatches(features_query,features_data,'fix',0.32);
% % [id_query, id_data] = getMatches(features_query,features_data,'KNN');
[id_query, id_data] = getMatches(features_query,features_data,'dration',0.95);
% 
figure; ax = axes;
showMatchedFeatures(I_query,I_data,Psift_query(id_query,:),Psift_data(id_data,:),'montage','Parent',ax);
title(ax, 'Identified Matches');
legend(ax, 'Matched points Query','Matched points Database');
x=1;

%% Part 3 - SURF
MetricThreshold = 6000;
rot_step = 15;
scale_step = 1.2;
% 
Surf_obj= Surf_det(MetricThreshold, rot_step, scale_step);
% 
I_query = imread('data1/obj1_5.JPG');
I_query = rgb2gray(I_query);
I_query=im2single(I_query);
% 
I_data = imread('data1/obj1_t1.JPG');
I_data = rgb2gray(I_data);
I_data=im2single(I_data);

Psurf_query = Surf_obj.extract_points(I_query);
[features_query,Psurf_query] = extractFeatures(I_query,Psurf_query,'Method','SURF');

Psurf_data = Surf_obj.extract_points(I_data);
[features_data,Psurf_data] = extractFeatures(I_data,Psurf_data,'Method','SURF');


[id_query, id_data] = getMatches(features_query,features_data,'fix',0.15);
% [id_query, id_data] = getMatches(features_query,features_data,'KNN');
% [id_query, id_data] = getMatches(features_query,features_data,'dration',0.7);
% 
figure; ax = axes;
showMatchedFeatures(I_query,I_data,Psurf_query(id_query,:),Psurf_data(id_data,:),'montage','Parent',ax);
title(ax, 'Identified Matches');
legend(ax, 'Matched points Query','Matched points Database');

function [id_query, id_data] = getMatches(feature_query,feature_data,mode,thold)
    id_query = (1:size(feature_query,1))';

    if mode == "KNN"
        D = pdist2(feature_query,feature_data);
        [~ , id_data] = min(D,[],2);
    elseif mode == "fix"
        % fix threshhold
        D = pdist2(feature_query,feature_data);
        [mn , id_data] = min(D,[],2);
        
        id_query = id_query(mn<thold);
        id_data = id_data(mn<thold);
    else       
        % ration threshhold
        D = pdist2(feature_query,feature_data);
        [mn_1 , id_data] = min(D,[],2);
        idx = sub2ind(size(D), id_query, id_data); % Convert To Linear Indexing
        D(idx) = inf;
        [mn_2 , ~] = min(D,[],2);
        ratio_metric = mn_1./mn_2;
        
        id_query = id_query(ratio_metric<thold);
        id_data = id_data(ratio_metric<thold);
    end

end

