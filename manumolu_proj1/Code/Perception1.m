%Reading the Video, finding the duration
v = VideoReader('project_video.mp4');
D = v.Duration;
numFrames = 0;
xy_longl= [0,0;0,0];
xy_longr= [0,0;0,0];


%Main Loop, Video running as series as Images
while hasFrame(v)
   video = readFrame(v);
   numFrames = numFrames+1;                                %Frame Count
   graysc_video = rgb2gray(video);                         %Converting to Grayscale
   denoised_video = medfilt2(graysc_video);                %Denoising the Image
   BW = edge(denoised_video, 'Canny',0.5);                 %Edge Detection
   x= [0 1280 1280 0];                                     % Mask Dimensions
   y =[360 360 720 720];
   
   %Mask = poly2mask(x,y,720,1280);                        %Generating Mask 
%   Using this to create mask
%   imshow(video)
%   Mask = roipoly(video) 
%( Use in this file or Seperate file to generate hand-drawn mask)
  
   Bottom_only = BW & Mask;                                %Concatenating Mask and Images
   [H,T,R] = hough(Bottom_only);                           %Hough Transform
   P= houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
   lines = houghlines(BW,T,R,P,'FillGap',10,'MinLength',5); %Hough Lines
   imshow(video), hold on
   
   max_len1=0;
   max_len2=0;
   
   bottom=720;
   top=450;
  
  for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   %plot(xy(:,1),xy(:,2),'LineWidth',0.01,'Color','green');
   
   
     if lines(k).theta >0
   % Determine the endpoints of the longest line segment (Left)
   len = norm(lines(k).point1 - lines(k).point2);
        if ( len > max_len1) 
        max_len1 = len;
        xy_longl = xy;
        end
     end
   
    s1 = (xy_longl(2,1)- xy_longl(1,1))/(xy_longl(2,2)-xy_longl(1,2));
   
    X1 = ((top-xy_longl(1,2))*s1) + xy_longl(1,1);
    X2 = ((bottom-xy_longl(1,2))*s1) + xy_longl(1,1);
   
    
       if lines(k).theta <0
        % Determine the endpoints of the longest line segment (Right)
        len = norm(lines(k).point1 - lines(k).point2);
            if ( len > max_len2) 
            max_len2 = len;
            xy_longr = xy;
            end
       end
  
    s2 = (xy_longr(2,1)- xy_longr(1,1))/(xy_longr(2,2)-xy_longr(1,2));
    
    X3 = ((top-xy_longr(1,2))*s2) + xy_longr(1,1);
    X4 = ((bottom-xy_longr(1,2))*s2) + xy_longr(1,1);
    
    
    
  end
  
    M1 = [ X1 top; X2 bottom] ;
    plot(M1(:,1),M1(:,2),'LineWidth',5,'Color','Red');
    
    M1 = [ X3 top; X4 bottom] ;
    plot(M1(:,1),M1(:,2),'LineWidth',5,'Color','Red');
    
    drawnow
    
    
  
end






