classdef Sift_det

    properties
        PeakThresh
        EdgeThresh
    end

    methods

        function obj = Sift_det(PeakThresh, EdgeThresh)
            obj.PeakThresh = PeakThresh;
            obj.EdgeThresh = EdgeThresh;
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

            
    end
end

