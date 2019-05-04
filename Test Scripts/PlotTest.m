 Xcube = [0;1;1;0;0];
 Ycube = [0;0;1;1;0];
 Zcube = [0;0;0;0;0];
 
 format long g
 P = [0,0,0;
      1,0,0;
      1,1,0;
      0,1,0;
      0,0,1;
      1,0,1;
      1,1,1;
      0,1,1];
 
X = P';  % Matrix containing information regarding all vertices. 
  
  
  
% Rotation Matrix for X-axis rotation of 20 Degrees
R =[1.0000000, 0.0000000,  0.0000000;
    0.0000000,  0.9396926, -0.3420202;
    0.0000000,  0.3420202,  0.9396926 ];

K = [800,0,250;
     0,800,250;
     0,0,1];

 figure;
 hold on;
 plot3(Xcube,Ycube,Zcube);   % Drawing original Cube
 plot3(Xcube,Ycube,Zcube+1); 


  for k=1:length(Xcube)-1
     plot3([Xcube(k);Xcube(k)],[Ycube(k);Ycube(k)],[0;1]);
  end


x = project(X,R,5,K);
Pnew = x';


XCubeNew1 = [Pnew(1:4),Pnew(1,1)];
YCubeNew1 = [Pnew(1:4, 2)' , Pnew(1,2)];
ZCubeNew1 = [Pnew(1:4, 3)' , Pnew(1,3)];

XCubeNew2 = [Pnew(5:8),Pnew(5,1)];
YCubeNew2 = [Pnew(5:8, 2)' , Pnew(5,2)];
ZCubeNew2 = [Pnew(5:8, 3)' , Pnew(5,3)];

 figure;
 hold on;
 plot3(XCubeNew1,YCubeNew1,ZCubeNew1);   % Drawing transformed Cube
 plot3(XCubeNew2,YCubeNew2,ZCubeNew2); 


  for k=1:4
     plot3([XCubeNew1(k);XCubeNew2(k)],[YCubeNew1(k);YCubeNew2(k)],[ZCubeNew1(k);ZCubeNew2(k)]);
  end


     function x = project(X,R,T,K);
     Size = size(X,2);
     
     x = K*(R*X + T*[zeros(2,Size);ones(1,Size)]);
     end
     
     
     
   
  
