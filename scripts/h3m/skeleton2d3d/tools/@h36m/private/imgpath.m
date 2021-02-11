function [ pa ] = imgpath( obj, idx, cam )

pa = sprintf('%02d/%02d/%1d/%1d_%04d.jpg', ...
    obj.ind2sub(idx,1),obj.ind2sub(idx,2), ...
    obj.ind2sub(idx,3),cam,obj.ind2sub(idx,4));

end

