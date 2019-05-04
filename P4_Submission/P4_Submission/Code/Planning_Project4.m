tic
%---> Takes a long time, so input a frame of your choice to see subsequent
%figures or otherwise comment out every figure() till the last one 
%Initialize
clear all;
close all;
clc;


%Need vlfeat toolbox installed either from add-ons or 
%from http://www.vlfeat.org/install-matlab.html
run('Add-Ons\Toolboxes\vlfeat-0.9.21\toolbox\vl_setup.m')

%Building Classifier
classifier = TrafficSignSVM();

%---------------------UNCOMMENT HERE TO BUILD VIDEO------------------%
% v = VideoWriter('TSR_Output_2','MPEG-4');
% v.FrameRate = 30;
% open(v);
%Need Input FOlder TSR Added to path 
% input_image_folder = ('C:\Users\Manohar Anumolu\Documents\MATLAB\TSR\input');
% image_count=[];
% for i = 32640:35500
% 
% filename = [sprintf('image.%06d',i) '.jpg'];
% 
% Input_File = fullfile(input_image_folder,filename);
%--------------------------------------------------------------------%

%----------Testing for Various Traffic Signs------------%
%----Used to check to calibrate MSER Parameters for Part1
% Input_File = 'TSR/input/image.033706.jpg'; %-----> Red Triangular Sign
% Input_File = 'TSR/input/image.033654.jpg'; %----> Arrow Sign (Blue)
% Input_File = 'TSR/input/image.034830.jpg'; %----> SpeedBreaker Sign (Red)
Input_File = 'TSR/input/image.033432.jpg'; %--Stop Sign (Red) delta=10
% Input_File = 'TSR/input/image.032850.jpg'; %---Parking Sign(Blue)
% Input_File = 'TSR/input/image.033537.jpg'; %---Cycle Sign
% Input_File = 'TSR/input/image.034773.jpg'; %---Road Narrowing Sign (does
% not get detected until score>-0.005!!!, Leads to Lot of False Positives
% on other Images);
% Input_File = 'TSR/input/image.034868.jpg'; %---Inverse Traingular Sign
% %Multiple Detections from multiple region blobs------------
% Input_File = 'TSR/input/image.033597.jpg'; %---Red Color Sign (which sign is it?)

%Read Input File(s)
img = imread(Input_File);
img = im2double(img);%------------------> Converting to Double
figure;
imshow(img);
title('Original Image');


%Denoise Image,sigma =2
img= imgaussfilt(img,2);
% figure;
% imshow(img);
% title('Filtered Image');

%Channel Normalization
[img,R,G,B] = RGB_Normalize(img);
figure;
imshow(img);
title('Contrast Adjusted Image');

%HSV Thresholding 
hsvRed = hsvdetectR(img);
hsvBlue = hsvdetectB(img);
hsvRB = hsvRed&hsvBlue;
figure;
imshow(hsvRed);
title('RED HSV');
figure;
imshow(hsvBlue);
title('BLUE HSV');
figure;
imshow(hsvRB);
title('RB HSV');

%Image Enhancing Constants
K_R = min(R-B,R-G)./(R+G+B);
K_B = (B-R)./(R+G+B);

%Image Enhancing and Contrast Normalization given by Satli et al 
Red = max(0,K_R);
Red = imgaussfilt(Red,2);
Low_High_Red = stretchlim(Red);
I_Red = imadjust(Red,Low_High_Red);
% figure;
% imshow(Red);
% title('Red Enhanced Image');

Blue = max(0,K_B);
Blue = imgaussfilt(Blue,2);
Low_High_Blue = stretchlim(Blue);
I_Blue = imadjust(Blue,Low_High_Blue);
% figure;
% imshow(Blue);
% title('Blue');

%Remove Bottom Half for both Red and Blue-------> Input for MSER
%Input is Double, vlfeat takes only uint8, transform used in MSER function
I_Red = RemoveBottomHalf(I_Red);
I_Blue = RemoveBottomHalf(I_Blue);
I_RB = I_Red + I_Blue;
figure;
imshow( I_Red);
title('MSER Input I RED');
figure;
imshow(I_Blue);
title('MSER Input I Blue');
figure;
imshow(I_RB);
title('MSER Input Red and Blue');



%------------MSER--------------%
[MRed,MRed1,regionsR] = MSER(I_Red);
[MBlue,MBlue1,regionsB] = MSER(I_Blue);
[MRB,MRB1,regionsRB] = MSER(I_RB);

[MRed,MRed1,regionsRwhsv] = MSER(I_Red&hsvRed);
[MBlue,MBlue1,regionsBwhsv] = MSER(I_Blue&hsvBlue);
[MRB,MRB1,regionsRBwhsv] = MSER(I_RB&hsvRB);


figure;
imagesc(I_Red) ; hold on ; axis equal off; colormap gray ;
[~,hR]=contour(MRed,(0:max(MRed(:)))+.5) ;
set(hR,'color','y','linewidth',3) ;
title('Red MSER');
figure;
imshow(MRed1);
title('RED MSER MASK');



figure;
imagesc(I_Blue) ; hold on ; axis equal off; colormap gray ;
[~,hB]=contour(MBlue,(0:max(MBlue(:)))+.5) ;
set(hB,'color','y','linewidth',3) ;
title('Blue MSER');
figure;
imshow(MBlue1);
title('Blue MSER MASK');

 
figure;
imagesc(I_RB) ; hold on ; axis equal off; colormap gray ;
[~,hRB]=contour(MRB,(0:max(MRB(:)))+.5) ;
set(hRB,'color','y','linewidth',3) ;
title('RB MSER');
figure;
imshow(MRB1);
title('RB MSER MASK');


%Get Bounding Boxes
% figure;
% clf; imagesc(img) ; hold on ; axis equal off; colormap gray ; 

AllFiles =[];
AllFilesBox =[];
TotalSigns=[];
if ~isempty(regionsB)
for k1 = 1 : length(regionsB)
    boxB = regionsB(k1).BoundingBox;
    ratioB = boxB(3)/boxB(4);
    if ratioB < 1.2 && ratioB > 0.6 %Aspect Ration of detections
    signB = imcrop(img, boxB);  
    signB = im2single(imresize(signB,[64 64]));
    AllFiles = cat(4,AllFiles,signB);
    AllFilesBox = cat(3,AllFilesBox,boxB);
%     filenameB = [sprintf('SignB_%03d',k1) '.jpg'];
%     filename = fullfile('P4_Submission','Output','SignsB',filenameB); %Building Paths for training Set
%     imwrite(signB,filename);
%     signB=[];
    end
end
end

if ~isempty(regionsR)
for k2 = 1 : length(regionsR)
    boxR = regionsR(k2).BoundingBox;
    ratioR = boxR(3)/boxR(4);
    if ratioR < 1.2 && ratioR > 0.6 %Aspect Ration of detections
    signR = imcrop(img, boxR);  
    signR = im2single(imresize(signR,[64 64]));
    AllFiles = cat(4,AllFiles,signR);
    AllFilesBox = cat(3,AllFilesBox,boxR);
%     filenameR = [sprintf('SignR_%03d',k2) '.jpg'];
%     filename = fullfile('P4_Submission','Output','SignsR',filenameR); %Building Paths for training Set
%     imwrite(signR,filename);
%     signR=[];
    end
end
end

if ~isempty(regionsRB)
for k3 = 1 : length(regionsRB)
    boxRB = regionsRB(k3).BoundingBox;
    ratioRB = boxRB(3)/boxRB(4);
    if ratioRB < 1.2 && ratioRB > 0.6 %Aspect Ration of detections
    signRB = imcrop(img, boxRB);  
    signRB = im2single(imresize(signRB,[64 64]));
    AllFiles = cat(4,AllFiles,signRB);
    AllFilesBox = cat(3,AllFilesBox,boxRB);
%     filenameRB = [sprintf('SignRB_%03d',k3) '.jpg'];
%     filename = fullfile('P4_Submission','Output','SignsRB',filenameRB); %Building Paths for training Set
%     imwrite(signRB,filename);
%     signRB=[];
    end
end
end

if ~isempty(regionsBwhsv)
for k4 = 1 : length(regionsBwhsv)
    boxBwhsv = regionsBwhsv(k4).BoundingBox;
    ratioBwhsv = boxBwhsv(3)/boxBwhsv(4);
    if ratioBwhsv < 1.2 && ratioBwhsv > 0.6 %Aspect Ration of detections
    signBwhsv = imcrop(img, boxBwhsv);  
    signBwhsv = im2single(imresize(signBwhsv,[64 64]));
    AllFiles = cat(4,AllFiles,signBwhsv);
    AllFilesBox = cat(3,AllFilesBox,boxBwhsv);
%     filenameBwhsv = [sprintf('SignBwhsv_%03d',k1) '.jpg'];
%     filename = fullfile('P4_Submission','Output','SignsBwhsv',filenameBwhsv); %Building Paths for training Set
%     imwrite(signB,filename);
%     signB=[];
    end
end
end

if ~isempty(regionsRwhsv)
for k5 = 1 : length(regionsRwhsv)
    boxRwhsv = regionsRwhsv(k5).BoundingBox;
    ratioRwhsv = boxRwhsv(3)/boxRwhsv(4);
    if ratioRwhsv < 1.2 && ratioRwhsv > 0.6 %Aspect Ration of detections
    signRwhsv = imcrop(img, boxRwhsv);  
    signRwhsv = im2single(imresize(signRwhsv,[64 64]));
    AllFiles = cat(4,AllFiles,signRwhsv);
    AllFilesBox = cat(3,AllFilesBox,boxRwhsv);
%     filenameR = [sprintf('SignR_%03d',k2) '.jpg'];
%     filename = fullfile('P4_Submission','Output','SignsR',filenameR); %Building Paths for training Set
%     imwrite(signR,filename);
%     signR=[];
    end
end
end

if ~isempty(regionsRBwhsv)
for k6 = 1 : length(regionsRBwhsv)
    boxRBwhsv = regionsRBwhsv(k6).BoundingBox;
    ratioRBwhsv = boxRBwhsv(3)/boxRBwhsv(4);
    if ratioRBwhsv < 1.2 && ratioRBwhsv > 0.6 %Aspect Ration of detections
    signRBwhsv = imcrop(img, boxRB);  
    signRBwhsv = im2single(imresize(signRBwhsv,[64 64]));
    AllFiles = cat(4,AllFiles,signRBwhsv);
    AllFilesBox = cat(3,AllFilesBox,boxRB);
%     filenameRB = [sprintf('SignRB_%03d',k3) '.jpg'];
%     filename = fullfile('P4_Submission','Output','SignsRB',filenameRB); %Building Paths for training Set
%     imwrite(signRB,filename);
%     signRB=[];
    end
end
end


%Stored All Classifier Input Blobs into AllFiles, a 4-D Array. 
%Iteratively Construct back the blob images to use as input.
if ~isempty(AllFiles)
if ~isempty(AllFilesBox)
for ii = 1:size(AllFiles,4)
%     figure;
    R1 = AllFiles(:,:,1,ii);
    B1 = AllFiles(:,:,2,ii);
    G1 = AllFiles(:,:,3,ii);
    
    box = AllFilesBox(:,:,ii);
    
    II(:,:,1) = R1;
    II(:,:,2) = B1;
    II(:,:,3) = G1;
   
%     imshow(II);
%     R1=[];
%     B1=[];
%     G1=[];
%     II=[]; 
    
    %Classification
    %Get HOG Features
    hog = [];
    hog_det = vl_hog(II, 4);
    [hog_1, hog_2] = size(hog_det);
    dim = hog_1 * hog_2;
    hog_det_trans = permute(hog_det, [2 1 3]);
    hog_det=reshape(hog_det_trans,[1 dim]); 
    hog = hog_det;
    [predictedLabels, score] = predict(classifier, hog);
    

     for j = 1:length(score)
         if (score(j) > -0.043)
             rectangle('Position', box,'EdgeColor','g','LineWidth',2)
             Sign_Name = strcat(predictedLabels,'.png');
             %Signs taken from:-
             %http://btsd.ethz.ch/shareddata/
             sign_path = fullfile('Signs',Sign_Name);
             

             Im_Sign2 = imread(sign_path);
             Im_Sign= im2single(imresize(Im_Sign2,[box(4) box(3)]));
%              width = BoundingBox(3);
%              height = BoundingBox(4);

%Output Figure-------------------------------------------------------->
             figure;
             imshow(img);
             title('TSR Image');
             hold on
             image([int64(box(1)-box(4)) int64(box(1)-box(4)) ],[int64(box(2)) int64(box(2))],Im_Sign);

         end
     end
     
 %--------------  Image will be displayed here every frame-------------------%          
end
end
end

%-------------------------------TO BUILD VIDEO-------
% frame = getframe(gca);
% writeVideo(v,frame);
% image_count=image_count+1;
% 
% pause(0.001);
% cla
% disp(image_count);
% end
% close(v);
% toc
%------------------------------------------------------------

function [IMG,R,G,B] = RGB_Normalize(A)
%Input- Image
%Output- Contrast Image and the three channels
R_Channel = A(:,:,1);
G_Channel = A(:,:,2);
B_Channel = A(:,:,3);

Low_HighR = stretchlim(R_Channel);
Low_HighG = stretchlim(G_Channel);
Low_HighB = stretchlim(B_Channel);

R = imadjust(R_Channel,Low_HighR);
G = imadjust(G_Channel,Low_HighG);
B = imadjust(B_Channel,Low_HighB);

IMG(:,:,1) = R;
IMG(:,:,2) = G;
IMG(:,:,3) = B;

end

function [I_Cropped] = RemoveBottomHalf(img)
%Input- img
%Output- Top half of img
X = [1 1628 1628 1 1];
Y = [1 1 618 618 1];
Mask = poly2mask(X,Y,1236,1628);
I_Cropped = immultiply(img,Mask);
end

function [hsvRed] = hsvdetectR(img)
%Detect Red in hsv
%Find HSV of image and take conservative lower bounds using Data Cursor
%after imshow
hsv = rgb2hsv(img);
h = hsv(:,:,1);
s = hsv(:,:,2);
v = hsv(:,:,3);
h_range = h >= 0 & h <=1;
s_range = s >= 0.5 & s <=0.9;
v_range = v >= 0.2 & v <=0.7;
hsvRed = h_range & s_range & v_range;
end

function [hsvBlue] = hsvdetectB(img)
%Detect Blue in hsv
%https://www.mathworks.com/matlabcentral/answers/324036-blue-colour-range-in-hsv-saturation-system
%https://www.mathworks.com/matlabcentral/answers/48053-how-to-enhance-blue-color-alone-of-an-image
%Find HSV of image and take conservative lower bounds using Data Cursor
%after imshow
hsv = rgb2hsv(img);
h = hsv(:,:,1);
s = hsv(:,:,2);
v = hsv(:,:,3);
h_range = h >= 0.5 & h <=1;
s_range = s >= 0.4 & s <= 0.85; 
v_range = v >= 0.3 & v <= 1; 
hsvBlue = h_range & s_range & v_range;
end

function [M,M1,regions] = MSER(img)
%Using vl-feat toolbox
%Input-Cropped,Contrast-Normalized uint8 image
%Output- Image, BOunding Box Regions
% extracting mser regions 
% Format--->
% I = uint8(rgb2gray(I)):
% Code from http://www.vlfeat.org/overview/mser.html
% (given in pipeline)
img = im2uint8(img);
[r,~] = vl_mser(img,'MinDiversity',0.7,'MaxVariation',0.2,'Delta',8);


%Plotting MSERS
M = zeros(size(img)) ;
for x=r'
 s = vl_erfill(img,x) ;
 M(s) = M(s) + 1;
end
  
thresh = graythresh(M);
M1 = im2bw(M, thresh);
% M1 = medfilt2(M1,[5 5]);
% se = strel('octagon',6);
% se1 = strel('Rectangle',[2,2]);
% M1 = imdilate(M1,se);
% M1 = imdilate(M1,se1);
%------------
% M1 = medfilt2(M1,[5 5]);
% se3 = strel('rectangle', [2 2]);  
% M1 = imerode(M1,se3);
M1 = imfill(M1,'holes');
%-------------
M1 = bwareafilt(M1, [500 10000]);
regions = regionprops(M1, 'BoundingBox');

end

function [classifier] = TrafficSignSVM()
%Classifier
training_folder = fullfile('TSR','Training');
trainingSet = imageSet(training_folder, 'recursive');
trainingFeatures = [];
trainingLabels   = [];
testingFeatures = [];
testingLabels   = [];

for i = 1:numel(trainingSet)
   numImages = trainingSet(i).Count;
   HOG = [];
   
   for j = 1:numImages
       I = read(trainingSet(i), j);
       %Resize Image to 64x64
       I = im2single(imresize(I,[64 64]));
       %Get HOG Features with cell size 
%      HOG_Features = vl_hog(I,2);
       HOG_Features = vl_hog(I,4);
%      HOG_Features = vl_hog(I,2);
%      HOG_Features = vl_hog(I,8);
       [HOG_R, HOG_C] = size(HOG_Features);
       Length = HOG_R*HOG_C;
       %Interchange rows and columns using permute
       HOG_T = permute(HOG_Features, [2 1 3]);
       HOG_Features=reshape(HOG_T,[1 Length]); 
       
       HOG(j,:) = HOG_Features;
   end
   
   labels = repmat(trainingSet(i).Description, numImages, 1);
   trainingFeatures = [trainingFeatures; HOG];
   trainingLabels = [trainingLabels; labels];
       
end

%SVM
classifier = fitcecoc(trainingFeatures, trainingLabels);
end

