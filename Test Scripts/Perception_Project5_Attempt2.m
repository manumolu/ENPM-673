%Initialize
clear all;
close all;
clc;

%Need vlfeat toolbox installed either from add-ons or 
%from http://www.vlfeat.org/install-matlab.html
run('Add-Ons\Toolboxes\vlfeat-0.9.21\toolbox\vl_setup.m')

%----------Testing for Various Traffic Signs------------%
% Input_File = 'TSR/input/image.033400.jpg'; %-----> Blue Parking Sign
% Input_File = 'TSR/input/image.033706.jpg'; %-----> Red Triangular Sign
Input_File = 'TSR/input/image.033654.jpg'; %----> Arrow Sign (Blue)
% Input_File = 'TSR/input/image.034830.jpg'; %----> SpeedBreaker Sign (Red)
% Input_File = 'TSR/input/image.033432.jpg'; %--Stop Sign (Red) delta=10
% Input_File = 'TSR/input/image.032850.jpg'; %---Parking Sign(Blue)
% Input_File = 'TSR/input/image.033597.jpg'; %---Red Color Sign (which sign is it?)

%Read Input File(s)
img = imread(Input_File);
img = im2double(img);%------------------> Converting to Double
% figure;
% imshow(img);

%Denoise Image,sigma =2
img= imgaussfilt(img,2);
% figure;
% imshow(img);
% title('Filtered Image');

%Contrast Stretching for all Channels
R_Channel = img(:,:,1);
G_Channel = img(:,:,2);
B_Channel = img(:,:,3);

Low_HighR = stretchlim(R_Channel);
Low_HighG = stretchlim(G_Channel);
Low_HighB = stretchlim(B_Channel);

R_Adjust = imadjust(R_Channel,Low_HighR);
G_Adjust = imadjust(G_Channel,Low_HighG);
B_Adjust = imadjust(B_Channel,Low_HighB);

%Contrast Adjusted Image 'C'
C(:,:,1) = R_Adjust;
C(:,:,2) = G_Adjust;
C(:,:,3) = B_Adjust;
figure;
imshow(C);
title('Contrast Adjusted Image');

R = C(:,:,1);
G = C(:,:,2);
B = C(:,:,3);


%Image Enhancing Constants
K_R = min(R-B,R-G)./(R+G+B);
K_B = (B-R)./(R+G+B);

% % Image Enhancing Constants
% K_R = min(R-B, R-G);
% K_B = (B-R);

%Image Enhancing given by Satli et al 
Red = max(0,K_R);
Red = imgaussfilt(Red,2);
% figure;
% imshow(Red);
% title('Red');

Blue = max(0,K_B);
Blue = imgaussfilt(Blue,2);
% figure;
% imshow(Blue);
% title('Blue');

%Contrast Normalization Again (Salti et al)
Low_High_Red = stretchlim(Red);
I_Red = imadjust(Red,Low_High_Red);
% figure;
% imshow(I_Red);
% title('IRed');

Low_High_Blue = stretchlim(Blue);
I_Blue = imadjust(Blue,Low_High_Blue);
% figure;
% imshow(I_Blue);
% title('IBue');

RB = Red+Blue;
% figure;
% imshow(RB);
% title('RB');

%Mask
X = [1 1628 1628 1 1];
Y = [1 1 618 618 1];
Mask = poly2mask(X,Y,1236,1628);

% Crop_Red
I_Red = immultiply(I_Red,Mask);
% figure;
% imshow(I_Red);
% title('IRed');

% Crop_Blue;
I_Blue = immultiply(I_Blue,Mask);
% figure;
% imshow(I_Blue);
% title('IBlue');

% CropRB
I_RB = immultiply(RB,Mask);
% figure;
% imshow(I_RB);
% title('Red+Blue');


%------------MSER--------------%
% extracting mser regions 
% Format
% I = uint8(rgb2gray(I)):
% Code from http://www.vlfeat.org/overview/mser.html

I_Red = im2uint8(I_Red);
I_Blue = im2uint8(I_Blue);
I_RB = im2uint8(I_RB);


%MSER for Red
[r,f] = vl_mser(I_Red,'MinDiversity',0.7,'MaxVariation',0.2,'Delta',8); 

MRed = zeros(size(I_Red));
for x=r'
 s = vl_erfill(I_Red,x);
 MRed(s) = MRed(s) + 1;
end
clf ; imagesc(I_Red); hold on ; axis equal off; colormap gray ;
[c,h]=contour(MRed,(0:max(MRed(:)))+.5);
set(h,'color','y','linewidth',3);
% figure;
% imshow(MRed);
% title('MRed')
thresh = graythresh(MRed);
MRed = im2bw(MRed, thresh);
se = strel('octagon',6);
MRed = imdilate(MRed,se);
MRed = bwareafilt(MRed, [950 10000]);
figure;
imshow(MRed)
title('Mred after');

%MSER for Blue
[r1,f1] = vl_mser(I_Blue,'MinDiversity',0.7,'MaxVariation',0.2,'Delta',8); 

MBlue = zeros(size(I_Blue));
for x1=r1'
 s1 = vl_erfill(I_Blue,x1);
 MBlue(s1) = MBlue(s1) + 1;
end
clf ; imagesc(I_Blue); hold on ; axis equal off; colormap gray ;
[c1,h1]=contour(MBlue,(0:max(MBlue(:)))+.5);
set(h1,'color','y','linewidth',3);
% figure;
% imshow(MBlue);
% title('MBlue')
thresh1 = graythresh(MBlue);
MBlue = im2bw(MBlue, thresh1);
se1 = strel('octagon',6);
MBlue = imdilate(MBlue,se1);
MBlue = bwareafilt(MBlue, [950 10000]);
figure;
imshow(MBlue)
title('MBlue after');

%MSER for RB
[r2,f2] = vl_mser(I_RB,'MinDiversity',0.7,'MaxVariation',0.2,'Delta',8); 

MRB = zeros(size(I_RB));
for x2=r2'
 s2 = vl_erfill(I_RB,x2);
 MRB(s2) = MRB(s2) + 1;
end
clf ; imagesc(I_RB); hold on ; axis equal off; colormap gray ;
[c2,h2]=contour(MRB,(0:max(MRB(:)))+.5);
set(h2,'color','y','linewidth',3);
% figure;
% imshow(MRB);
% title('MRB')
thresh2 = graythresh(MRB);
MRB = im2bw(MRB,thresh2);
se2 = strel('octagon',6);
MRB = imdilate(MRB,se2);
MRB = bwareafilt(MRB, [950 10000]);
figure;
imshow(MRB)
title('MRB after');




































