clear all;
close all;
clc;
 Input_File = 'TSR/input/image.033400.jpg'; %-----> Blue Parking Sign
% Input_File = 'TSR/input/image.033706.jpg'; %-----> Red Triangular Sign
% Input_File = 'TSR/input/image.033654.jpg'; %----> Arrow Sign (Blue)
%  Input_File = 'TSR/input/image.034830.jpg'; %----> SpeedBreaker Sign (Red)
% Input_File = 'TSR/input/image.033432.jpg'; %--Stop Sign (Red) delta=10
% Input_File = 'TSR/input/image.032850.jpg'; %---Parking Sign(Blue)
% Input_File = 'TSR/input/image.033597.jpg'; %---Red Color Sign
I = imread(Input_File);
imshow(I);

red_mask = threshold_red(I);
blue_mask = threshold_blue(I);
    
% figure;
% imshow(red_mask);
% figure;
% imshow(blue_mask);


    
%Denoise the Image with sigma =2
I_filt = imgaussfilt(I,2);
%imshow(I_filt);

%Contrast Stretching
R_Channel = I_filt(:,:,1);
G_Channel = I_filt(:,:,2);
B_Channel = I_filt(:,:,3);

Low_HighR = stretchlim(R_Channel);
Low_HighG = stretchlim(G_Channel);
Low_HighB = stretchlim(B_Channel);

R_Adjust = imadjust(R_Channel,Low_HighR);
G_Adjust = imadjust(G_Channel,Low_HighG);
B_Adjust = imadjust(B_Channel,Low_HighB);

% imshow(R_Adjust);
% imshow(B_Adjust);
% imshow(G_Adjust);

%Adjusted Image
C(:,:,1) = R_Adjust;
C(:,:,2) = G_Adjust;
C(:,:,3) = B_Adjust;

R = double(C(:,:,1));
G = double(C(:,:,2));
B = double(C(:,:,3));
% figure;
% imshow(C);
% title('C');


%Image Enhancing Constants
K_R = min(R-B, R-G)./(R+G+B);
K_B = (B-R)./(R+G+B);

%Image Enhancing given by Satli et al 
Red = max(0,K_R);
Red = imgaussfilt(Red,2);
% Red = imfilter(Red,H);
% figure;
% imshow(Red);
% title('Red');

Blue = max(0,K_B);
Blue = imgaussfilt(Blue,2);
% Blue = imfilter(Blue,H);
% figure;
% imshow(Blue);
% title('Blue');

I_New_Red = Red;
I_New_Blue = Blue; 

Low_HighNew_Red = stretchlim(I_New_Red);
I_New_Red = imadjust(I_New_Red,Low_HighNew_Red);

Low_HighNew_Blue = stretchlim(I_New_Blue);
I_New_Blue = imadjust(I_New_Blue,Low_HighNew_Blue);

x = [1 1628 1628 1];
y = [1 1 618 618];
mask = poly2mask(x,y, 1236, 1628);

I_Crop_Red  = (immultiply(I_New_Red,mask));
I_Crop_Blue = (immultiply(I_New_Blue,mask));




figure;
imshow(I_Crop_Red & red_mask);
title('RED MSER')
hold on

[regions,cc] = detectMSERFeatures(I_Crop_Red,'ThresholdDelta',10.5,'MaxAreaVariation',0.1,'RegionAreaRange',[1000 5000]);
stats = regionprops('table',cc,'Eccentricity');
eccentricityIdx = stats.Eccentricity < 0.8;
circularRegions = regions(eccentricityIdx);
% plot(regions,'showPixelList',false,'showEllipses',true);
plot(circularRegions,'showPixelList',false,'showEllipses',true)
hold off



figure;
imshow(I_Crop_Blue & blue_mask);
title('BLUE MSER');
hold on;
[regions1,cc1] = detectMSERFeatures(I_Crop_Blue,'ThresholdDelta',10.5,'MaxAreaVariation',0.1,'RegionAreaRange',[950 5000]);
stats1 = regionprops('table',cc1,'Eccentricity');
eccentricityIdx1 = stats1.Eccentricity < 0.8;
circularRegions1 = regions1(eccentricityIdx1);
% plot(regions,'showPixelList',false,'showEllipses',true);
plot(circularRegions1,'showPixelList',false,'showEllipses',true)
hold off







function red_mask = threshold_red(im)
% Function to threshold an input image for Red coloured traffic signs in
% HSV space
% Output - red_mask -> BW image obtained after thresholding
im_hsv = rgb2hsv(im);
im_s = im_hsv(:,:,2);
im_v = im_hsv(:,:,3);
%imtool(im_v)
%imtool(im_s)
s_bin1 = im_s >= 0.5 & im_s <=0.9;
v_bin1 = im_v >= 0.20 & im_v <=0.65;
red_mask = s_bin1 & v_bin1;
end

function blue_mask = threshold_blue(im)
% Function to threshold an input image for blue coloured traffic signs in
% HSV space
% Output - blue_mask -> BW image obtained after thresholding
im_hsv = rgb2hsv(im);
im_s = im_hsv(:,:,2);
im_v = im_hsv(:,:,3);
im_s_bw = im_s >= 0.45 & im_s <= 0.8; % decreased lower bound from 0.6 to 0.35 to 0.45
im_v_bw = im_v >= 0.35 & im_v <= 1; % Could be done away with
blue_mask = im_s_bw & im_v_bw;
end


     






















