
H36MDataBase.instance();

Features{1} = H36MPose3DPositionsFeature();

vis_root = 'output/vis_3d_pose/';

figure(1);

if 1
    part = 'body';
else
    part = 'full';
end

for s = [1 5 6 7 8 9 11]
    for a = 2:16
        for b = 1:2
            tt = tic;
            fprintf('s/a/b %d/%02d/%d  ',s,a,b);
            vis_dir = [vis_root sprintf('%d-%02d-%d/',s,a,b)];
            makedir(vis_dir);
            
            c = 1;
            Sequence = H36MSequence(s, a, b, c);
            F = H36MComputeFeatures(Sequence, Features);
            Subject = Sequence.getSubject();
            posSkel = Subject.getPosSkel();
            [pose, posSkel] = Features{1}.select(F{1}, posSkel, part);
            
            minz = -200;
            maxz = 2000;
            minx = min(min(F{1}(:,1:3:94)));
            miny = min(min(F{1}(:,2:3:95)));
            maxx = max(max(F{1}(:,1:3:94)));
            maxy = max(max(F{1}(:,2:3:95)));
            m = max(ceil(abs([minx miny maxx maxy])/100)*100);
            support = [ ...
                -m -m m m -m; ...
                -m m m -m -m; ...
                0 0 0 0 0 ...
                ]';
            
            num_frame = size(F{1},1);
            for i = 1:num_frame
                if ismember(i,round((0.1:0.1:1.0) * num_frame))
                    fprintf('.');
                end
                vis_file = [vis_dir num2str(i,'%04d.png')];
                if exist(vis_file,'file')
                    continue
                end
                clf;
                % draw skeleton
                showPose(pose(i,:),posSkel);
                ylabel('y');
                zlabel('z');
                % draw ground plane
                hg = patch(support(:,1),support(:,2),support(:,3),[0 1 1]);
                alpha(hg,0.5);
                % plot origin world coordinates
                plot3(0,0,0,'o', ...
                    'MarkerEdgeColor','k',...
                    'MarkerFaceColor','k',...
                    'MarkerSize',10);
                % set axis range
                axis([-m m -m m minz maxz])
                % set viewpoint
                view([35,30]);
                % drawnow
                drawnow;
                % save figure
                print(gcf,vis_file,'-dpng','-r0');
            end
            
            vis_file = [vis_root sprintf('%d-%02d-%d.avi',s,a,b)];
            FrameRate = 50;
            if ~exist(vis_file,'file')
                % intialize video writer
                v = VideoWriter(vis_file);
                v.FrameRate = FrameRate;
                % open new video
                open(v);
                for i = 1:num_frame
                    % read image
                    file_im = [vis_dir num2str(i,'%04d.png')];
                    im = imread(file_im);
                    writeVideo(v,im);
                end
                % close video
                close(v);
            end
            
            time = toc(tt);
            fprintf('  %7.2f sec.\n',time);
        end
    end
end

close;
