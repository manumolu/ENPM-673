%Classifier
%Need vlfeat toolbox installed either from add-ons or 
%from http://www.vlfeat.org/install-matlab.html

%Only written to validate with testset
%Actual classifier for project implkemented as a function in the .m script.
run('Add-Ons\Toolboxes\vlfeat-0.9.21\toolbox\vl_setup.m')

training_folder = fullfile('TSR','Training');
trainingSet = imageSet(training_folder, 'recursive');

trainingFeatures = [];
trainingLabels   = [];

testingFeatures = [];
testingLabels   = [];

for i = 1:numel(trainingSet)
   ImTrainCount = trainingSet(i).Count;
   HOG = [];
   
   for j = 1:ImTrainCount
       I = read(trainingSet(i), j);
       %Resize Image to 64x64
       I = im2single(imresize(I,[64 64]));
       %Get HOG Features,cellsize
      %HOG_Features = vl_hog(I,2);
       HOG_Features = vl_hog(I,4);
      %HOG_Features = vl_hog(I,8);
       [HOG_R, HOG_C] = size(HOG_Features);
       Length = HOG_R*HOG_C;
       HOG_T= permute(HOG_Features, [2 1 3]);
       HOG_Features=reshape(HOG_T,[1 Length]); 
       HOG(j,:) = HOG_Features;
   end
   
   labels = repmat(trainingSet(i).Description, ImTrainCount, 1);
   trainingFeatures = [trainingFeatures; HOG];
   trainingLabels = [trainingLabels; labels];
       
end

%SVM
classifier = fitcecoc(trainingFeatures, trainingLabels);
clearvars -except classifier ;

% Testing the Classifier
%Evaluate the Classifier
testing_folder = fullfile('TSR','Testing');
testingSet = imageSet(testing_folder, 'recursive');


for ii = 1:numel(testingSet)
   Imtest_Count = testingSet(ii).Count;
   HOG_Test = [];
   
   for jj = 1:Imtest_Count
       I_Test = read(testingSet(ii), jj);
       %Resize Image to 64x64
       I_Test = im2single(imresize(I_Test,[64 64]));
       %Get HOG Features,cellsize
       %HOG_Features_Test = vl_hog(I_Test,2);
       HOG_Features_Test = vl_hog(I_Test,4);
       %HOG_Features_Test = vl_hog(I_Test,8);
       [HOG_R_Test, HOG_C_Test] = size(HOG_Features_Test);
       Length_Test = HOG_R_Test*HOG_C_Test;
       HOG_T_Test = permute(HOG_Features_Test, [2 1 3]);
       HOG_Features_Test=reshape(HOG_T_Test,[1 Length_Test]); 
       HOG_Test(jj,:) = HOG_Features_Test;
   end
   
   labels_test = repmat(testingSet(ii).Description, Imtest_Count, 1);
   testingFeatures = [testingFeatures; HOG_Test];
   testingLabels = [testingLabels; labels_test];
  
   [predictedLabels,score] = predict(classifier, testingFeatures);
   confMat = confusionmat(testingLabels, predictedLabels);
   

end
%http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology
   diag = trace(confMat);
   total = sum(sum(confMat));
   Accuracy = diag/total;
   disp(Accuracy);%--------->. Accuracy of Classifier



