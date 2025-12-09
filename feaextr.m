numSubjects = size(preprocessedGaitData, 1);
numFeatures = 6; 

featureMatrix = zeros(numSubjects, numFeatures);
labels = cell(numSubjects, 1);

for i = 1:numSubjects
    
    signal = preprocessedGaitData{i, 2}; 
    labels{i} = preprocessedGaitData{i, 3}; 

    if isempty(signal) || strcmp(labels{i}, 'Unknown')
        continue;
    end
        
    validSignal = signal(signal > 0);
    if isempty(validSignal), continue; end
   
    feat_1 = mean(validSignal);        
    feat_2 = std(validSignal) / feat_1; 

    if length(validSignal) > 1
        R = corrcoef(validSignal(1:end-1), validSignal(2:end)); 
        feat_3 = R(1, 2);
        if isnan(feat_3), feat_3 = 0; end
    else
        feat_3 = 0;
    end
    
    try
        [Pxx, F] = pwelch(validSignal, [], [], [], 100); 
        low_freq_power = sum(Pxx(F < 0.5)); 
        total_power = sum(Pxx);
        if total_power == 0
            feat_4 = 0;
        else
            feat_4 = low_freq_power / total_power; 
        end
    catch
        feat_4 = 0; 
    end

    N = length(validSignal); 
    mu = mean(validSignal); 
    sigma = std(validSignal);
    
    if sigma == 0
        feat_5 = 0;
        feat_6 = 0;
    else
        standardized_data = (validSignal - mu) / sigma;
        feat_5 = (1/N) * sum(standardized_data.^3); 
        feat_6 = (1/N) * sum(standardized_data.^4); 
    end
    
    featureMatrix(i, 1) = feat_1;
    featureMatrix(i, 2) = feat_2;
    featureMatrix(i, 3) = feat_3;
    featureMatrix(i, 4) = feat_4;
    featureMatrix(i, 5) = feat_5;
    featureMatrix(i, 6) = feat_6;
    
end

invalidRows = featureMatrix(:,1) == 0 | any(isnan(featureMatrix), 2);

featureMatrix(invalidRows, :) = [];
labels(invalidRows) = [];

emptyLabels = cellfun('isempty', labels);
featureMatrix(emptyLabels, :) = [];
labels(emptyLabels) = [];

isChar = cellfun(@ischar, labels);
featureMatrix(~isChar, :) = [];
labels(~isChar) = [];
