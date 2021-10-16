classdef Sift_det

    properties
        PeakThresh
        EdgeThresh
        rot_step
        scale_step
    end

    methods

        function obj = Sift_det(PeakThresh, EdgeThresh, rot_step, scale_step)
            obj.PeakThresh = PeakThresh;
            obj.EdgeThresh = EdgeThresh;
            obj.rot_step = rot_step;
            obj.scale_step = scale_step;
        end

        function [Psift,Fsift] = extract_points(obj,I)
            [points, features] = vl_sift(I,'PeakThresh',obj.PeakThresh,'EdgeThresh',obj.EdgeThresh);
            Psift = points(1:2,:)';
            Fsift = single(features);
            % normalizing features
            Fsift = (Fsift./vecnorm(Fsift))';
            
%             figure()
%             imshow(I)
%             hold on 
%             plot(Psift(:,1),Psift(:,2),'*g')
        end

       function repeatability = compute_repeatability_angle(obj,I,Psift)
            center_coordinates = [(size(I,2)-1)/2+1; (size(I,1)-1)/2 + 1];

            image_frame = [1, size(I,2), 1, size(I,2);...
                           1, 1, size(I,1), size(I,1)];
            repeatability = [];
            for angle=0:obj.rot_step:360   
                R = rotz(angle);
                rotated_image_frame = [-1;1].*(R(1:2,1:2) * ([-1;1].*(image_frame - center_coordinates))...
                                + [-1;1].*center_coordinates);                    
                                    
                translation_frame = min(rotated_image_frame')-1;                            
                I_temp = imrotate(I,angle,'loose');
                Psift_temp = obj.extract_points(I_temp);
                Psift_prime = ([-1;1].*(R(1:2,1:2) * ([-1;1].*(Psift' - center_coordinates))...
                                + [-1;1].*center_coordinates) - translation_frame')';
                               
                %%% computing repeatability
                Dx = pdist2(Psift_prime(:,1),Psift_temp(:,1));
                
                idx = Dx>2;

                Dy = pdist2(Psift_prime(:,2),Psift_temp(:,2));
                Dy(idx) = inf;
                
                [dy, ~] = min(Dy,[],2);

                current_rep = dy <= 2;
%                 plot(Psift_prime(current_rep==1,1),Psift_prime(current_rep==1,2),'Ob')
%                 plot(Psift_prime(current_rep~=1,1),Psift_prime(current_rep~=1,2),'Or')

                current_rep = mean(current_rep);
                repeatability(end+1) = current_rep;
            end
       end
        
       function repeatability = compute_repeatability_scale(obj,I,Psift)
            scales = power(obj.scale_step,0:8);
           
            repeatability = [];
            for scale=scales              
                
                Psift_prime = Psift * scale;
                
                I_temp = imresize(I,scale);
                Psift_temp = obj.extract_points(I_temp);
%                 plot(Psift_prime(:,1),Psift_prime(:,2),'or')
                
                %%% computing repeatability
                Dx = pdist2(Psift_prime(:,1),Psift_temp(:,1));
                
                idx = Dx>2;

                Dy = pdist2(Psift_prime(:,2),Psift_temp(:,2));
                Dy(idx) = inf;
                
                [dy, ~] = min(Dy,[],2);

                current_rep = dy <= 2;
%                 plot(Psift_prime(current_rep==1,1),Psift_prime(current_rep==1,2),'Ob')
%                 plot(Psift_prime(current_rep~=1,1),Psift_prime(current_rep~=1,2),'Or')

                current_rep = mean(current_rep);
                repeatability(end+1) = current_rep;
            end
        end

       
    end
end

