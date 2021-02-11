function [ coord_c ] = loadPoseCamera( obj, idx, cam )

coord_c = reshape(obj.coord_c(cam,idx,:),[3 size(obj.coord_c,3)/3])';
coord_c = double(coord_c);

end

