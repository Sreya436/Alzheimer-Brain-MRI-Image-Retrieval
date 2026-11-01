clc;
clear;
close all;

%% =========================================================
%  ALZHEIMER'S BRAIN MRI IMAGE RETRIEVAL
%  Pattern Similarity Based Medical Image Retrieval
%  MATLAB Implementation
% ==========================================================

%% 1. DATASET PATH
% CHANGE THIS PATH TO YOUR DATASET LOCATION

datasetPath = "C:\YourPath\Alzheimer Dataset\Original";

% Create image datastore
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

fprintf('\n============================================\n');
fprintf('ALZHEIMER MRI IMAGE RETRIEVAL SYSTEM\n');
fprintf('============================================\n');

fprintf('\nTotal number of images = %d\n', numel(imds.Files));

disp(' ');
disp('Images in each class:');
disp(countEachLabel(imds));


%% 2. PREPARE FEATURE DATABASE

numImages = numel(imds.Files);

% Each 256x256 image is divided into 32x32 blocks.
% 8 x 8 = 64 blocks
% Each block gives 4 GLCM features.
% Therefore:
% 64 x 4 = 256 features per image

featureDatabase = zeros(numImages,256);

fprintf('\n============================================\n');
fprintf('FEATURE EXTRACTION STARTED\n');
fprintf('============================================\n');


%% 3. FEATURE EXTRACTION

for i = 1:numImages

    % Read and preprocess image
    I = imread(imds.Files{i});

    % Convert RGB image to grayscale if necessary
    if size(I,3) == 3
        I = rgb2gray(I);
    end

    % Resize all images to 256 x 256
    I = imresize(I,[256 256]);

    % Convert image to uint8
    I = im2uint8(mat2gray(I));

    % Extract block-based GLCM features
    featureVector = extractGLCMFeatures(I);

    % Store feature vector
    featureDatabase(i,:) = featureVector;

    fprintf('Processing image %d / %d\n', ...
        i,numImages);

end


fprintf('\nFeature extraction completed.\n');

fprintf('Feature database size: %d x %d\n', ...
    size(featureDatabase,1), ...
    size(featureDatabase,2));


%% 4. SAVE FEATURE DATABASE

save('Alzheimer_FeatureDatabase.mat', ...
    'featureDatabase');

fprintf('\nFeature database saved successfully.\n');


%% 5. NORMALIZATION

% Standardize features
featureDatabase = zscore(featureDatabase);

% Replace possible NaN values with zero
featureDatabase(isnan(featureDatabase)) = 0;


%% 6. PCA DIMENSION REDUCTION

fprintf('\n============================================\n');
fprintf('PCA DIMENSION REDUCTION\n');
fprintf('============================================\n');

[coeff, score, ~, ~, explained] = ...
    pca(featureDatabase);

% Number of PCA components
numComponents = min(15,size(score,2));

reducedFeatures = score(:,1:numComponents);

fprintf('Original feature dimension = %d\n', ...
    size(featureDatabase,2));

fprintf('Reduced feature dimension = %d\n', ...
    numComponents);

fprintf('Variance explained by selected components = %.2f %%\n', ...
    sum(explained(1:numComponents)));


%% 7. EM-BASED CLUSTERING USING GMM

fprintf('\n============================================\n');
fprintf('EM-BASED CLUSTERING\n');
fprintf('============================================\n');

% Four classes in Alzheimer MRI dataset
numClusters = 4;

% EM optimization options
options = statset('MaxIter',500);

% Gaussian Mixture Model
gmmModel = fitgmdist( ...
    reducedFeatures, ...
    numClusters, ...
    'RegularizationValue',0.01, ...
    'Replicates',3, ...
    'Options',options);

% Assign each image to a cluster
clusterIndex = cluster(gmmModel,reducedFeatures);

fprintf('Number of clusters = %d\n',numClusters);

fprintf('EM clustering completed.\n');


%% 8. SAVE EM MODEL

save('Alzheimer_EM_Model.mat', ...
    'gmmModel', ...
    'coeff', ...
    'clusterIndex');

fprintf('EM model saved successfully.\n');


%% 9. DISPLAY SAMPLE MRI IMAGE

figure;

sampleImage = imread(imds.Files{1});

imshow(sampleImage);

title('Sample Alzheimer Brain MRI');


%% 10. SELECT QUERY IMAGE

fprintf('\n============================================\n');
fprintf('QUERY IMAGE\n');
fprintf('============================================\n');

% OPTION 1:
% Give the complete path of your query image here.

queryPath = "C:\YourPath\Alzheimer Dataset\Original\NonDemented\image.jpg";


%% 11. READ QUERY IMAGE

queryImage = imread(queryPath);

% Convert RGB to grayscale
if size(queryImage,3) == 3
    queryImage = rgb2gray(queryImage);
end

% Resize query image
queryImage = imresize(queryImage,[256 256]);

% Convert to uint8
queryImage = im2uint8(mat2gray(queryImage));


%% 12. EXTRACT QUERY IMAGE FEATURES

queryFeature = extractGLCMFeatures(queryImage);


%% 13. NORMALIZE QUERY FEATURES

% The query must be normalized using the same feature
% transformation used for the database.

queryFeature = zscore(queryFeature);

queryFeature(isnan(queryFeature)) = 0;


%% 14. PROJECT QUERY INTO PCA SPACE

queryReduced = queryFeature * coeff(:,1:numComponents);


%% 15. SIMILARITY / DISTANCE CALCULATION

fprintf('\n============================================\n');
fprintf('SIMILARITY CALCULATION\n');
fprintf('============================================\n');

distances = zeros(numImages,1);

for i = 1:numImages

    % Euclidean distance
    distances(i) = norm( ...
        queryReduced - reducedFeatures(i,:) );

end


%% 16. SORT IMAGES ACCORDING TO SIMILARITY

[sortedDistances, sortedIndex] = ...
    sort(distances,'ascend');


%% 17. TOP-K RETRIEVAL

topK = 5;

% Make sure topK does not exceed number of images
topK = min(topK,numImages);

topImages = sortedIndex(1:topK);


%% 18. DISPLAY RETRIEVED RESULTS

fprintf('\n============================================\n');
fprintf('TOP %d SIMILAR MRI IMAGES\n',topK);
fprintf('============================================\n');

figure;

for i = 1:topK

    subplot(1,topK,i);

    resultImage = imread(imds.Files{topImages(i)});

    imshow(resultImage);

    title(sprintf('Rank %d',i));

end


%% 19. DISPLAY DISTANCES

fprintf('\nSimilarity ranking:\n');

for i = 1:topK

    fprintf('Rank %d --> Distance = %.4f\n', ...
        i, ...
        sortedDistances(i));

end


%% 20. DISPLAY QUERY IMAGE + RESULTS

figure;

subplot(2,3,1);

imshow(queryImage);

title('Query MRI');


for i = 1:topK

    subplot(2,3,i+1);

    resultImage = imread(imds.Files{topImages(i)});

    imshow(resultImage);

    title(sprintf('Rank %d',i));

end


sgtitle('Alzheimer Brain MRI Image Retrieval');


%% 21. DISPLAY CLASS LABELS OF RETRIEVED IMAGES

fprintf('\n============================================\n');
fprintf('RETRIEVED IMAGE CLASSES\n');
fprintf('============================================\n');

for i = 1:topK

    imageIndex = topImages(i);

    fprintf('Rank %d --> Class: %s --> Distance: %.4f\n', ...
        i, ...
        string(imds.Labels(imageIndex)), ...
        sortedDistances(i));

end


fprintf('\n============================================\n');
fprintf('IMAGE RETRIEVAL COMPLETED\n');
fprintf('============================================\n');


%% =========================================================
% LOCAL FUNCTION
% BLOCK-BASED GLCM FEATURE EXTRACTION
% ==========================================================

function featureVector = extractGLCMFeatures(I)

    % Image size = 256 x 256
    blockSize = 32;

    % Initialize feature vector
    featureVector = [];

    % Divide image into 32 x 32 blocks
    for r = 1:blockSize:256

        for c = 1:blockSize:256

            % Extract block
            block = I( ...
                r:r+blockSize-1, ...
                c:c+blockSize-1);


            %% GLCM generation

            glcm = graycomatrix( ...
                block, ...
                'Offset',[0 1], ...
                'NumLevels',8, ...
                'Symmetric',true);


            %% GLCM properties

            stats = graycoprops( ...
                glcm, ...
                {'Contrast', ...
                 'Correlation', ...
                 'Energy', ...
                 'Homogeneity'});


            %% Add features to feature vector

            featureVector = [ ...
                featureVector, ...
                stats.Contrast, ...
                stats.Correlation, ...
                stats.Energy, ...
                stats.Homogeneity];

        end

    end

end