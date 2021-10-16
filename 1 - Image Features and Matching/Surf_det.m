classdef Surf_det

    properties
        MetricThreshold
        rot_step
        scale_step
    end

    methods

        function obj = Surf_det(MetricThreshold, rot_step, scale_step)
            obj.MetricThreshold = MetricThreshold;
            obj.rot_step = rot_step;
            obj.scale_step = scale_step;
        end

        function Psurf = extract_points(obj,I)
            points = detectSURFFeatures(I,'MetricThreshold',obj.MetricThreshold);
            Psurf = points.Location;
%             figure()
%             imshow(I)
%             hold on 
%             plot(Psurf(:,1),Psurf(:,2),'*g')
        end

       function repeatability = compute_repeatability_angle(obj,I,Psurf)
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
                Psurf_temp = obj.extract_points(I_temp);
                Psurf_prime = ([-1;1].*(R(1:2,1:2) * ([-1;1].*(Psurf' - center_coordinates))...
                                + [-1;1].*center_coordinates) - translation_frame')';
%                 plot(Psurf_prime(:,1),Psurf_prime(:,2),'or')
                
               %%% computing repeatability
                Dx = pdist2(Psurf_prime(:,1),Psurf_temp(:,1));
                
                idx = Dx>2;

                Dy = pdist2(Psurf_prime(:,2),Psurf_temp(:,2));
                Dy(idx) = inf;
                
                [dy, ~] = min(Dy,[],2);

                current_rep = dy <= 2;
%                 plot(Psurf_prime(current_rep==1,1),Psurf_prime(current_rep==1,2),'Ob')
%                 plot(Psurf_prime(current_rep~=1,1),Psurf_prime(current_rep~=1,2),'Or')

                current_rep = mean(current_rep);
                repeatability(end+1) = current_rep;
            end
       end
        
       function repeatability = compute_repeatability_scale(obj,I,Psurf)
            scales = power(obj.scale_step,0:8);
           
            repeatability = [];
            for scale=scales              
                
                Psurf_prime = Psurf * scale;
                
                I_temp = imresize(I,scale);
                Psurf_temp = obj.extract_points(I_temp);
%                 plot(Psurf_prime(:,1),Psurf_prime(:,2),'or')
                
                %%% computing repeatability
                Dx = pdist2(Psurf_prime(:,1),Psurf_temp(:,1));
                
                idx = Dx>2;

                Dy = pdist2(Psurf_prime(:,2),Psurf_temp(:,2));
                Dy(idx) = inf;
                
                [dy, ~] = min(Dy,[],2);

                current_rep = dy <= 2;
%                 plot(Psurf_prime(current_rep==1,1),Psurf_prime(current_rep==1,2),'Ob')
%                 plot(Psurf_prime(current_rep~=1,1),Psurf_prime(current_rep~=1,2),'Or')

                current_rep = mean(current_rep);
                repeatability(end+1) = current_rep;
            end
        end

       
    end
end

