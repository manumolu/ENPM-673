Input_File = 'TSR/input/image.033430.jpg';
I = imread(Input_File);

%Denoise the Image
I_filt = imgaussfilt(I,2);
% imshow(I)
figure;
% imshow(I_filt)

%Channel
R_Channel = I_filt(:,:,1);
G_Channel = I_filt(:,:,2);
B_Channel = I_filt(:,:,3);

% %Channel without Contrast Normalisation--Comment above and below immediate topics
% R = I_filt(:,:,1);
% G = I_filt(:,:,2);
% B = I_filt(:,:,3);

%Contrast Normalization
R = imadjust(R_Channel,stretchlim(R_Channel),[]);
% imshow(R_Channel)
% figure;
% imshow(R)

G = imadjust(G_Channel,stretchlim(G_Channel),[]);
% imshow(G_Channel)
% figure;
% imshow(G)

B = imadjust(B_Channel,stretchlim(B_Channel),[]);
% imshow(B_Channel)
% figure;
% imshow(B)

%Image enhancing given by Salti et al
Red_Sign = uint8(max(0, min(R-B, R-G)./R+B+G));
Blue_Sign = uint8(max(0, B-R./R+B+G));

% Red_Sign = uint8(max(0, min(R-B, R-G)));
% Blue_Sign = uint8(max(0, B-R));

% imshow(Red_Sign)
% figure;
% imshow(Blue_Sign)
% K_R = graythresh(Red_Sign);
% K_B = graythresh(Blue_Sign);
% BW_R = imbinarize(Red_Sign,K_R);
% BW_B = imbinarize(Blue_Sign,K_B);
% imshow(BW_R)
% figure;
% imshow(BW_B)

I_New = Red_Sign + Blue_Sign;
imshow(I_New)

%Mask to remove bottom 1/3rd of Image
x = [1 1628 1628 1 1];
y = [1 1 824 824 1];
mask = poly2mask(x,y,1236,1628);
I_Crop = uint8(immultiply(I_New,mask));
figure(2);
imshow(I_Crop);

%MSER
[label,n] = imser(I_Crop, 'light');
idisp(label)























