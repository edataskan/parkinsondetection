if (~exist('XTrain', 'var') || ~exist('YTrain', 'var')) 
    if exist('featureMatrix', 'var') && exist('labels', 'var')
        valid_idx = any(featureMatrix, 2) & ~cellfun(@isempty, labels);
        X_all = featureMatrix(valid_idx, :);
        Y_all = labels(valid_idx);
       
        Y_all = string(Y_all);
        Y_num = zeros(size(Y_all));
        Y_num(Y_all == "PD") = 1;
        Y_num(Y_all == "HC") = 0;
        
        try
            cv = cvpartition(Y_num, 'HoldOut', 0.30);
            trainIdx = training(cv);
            testIdx = test(cv);
            
            XTrain = X_all(trainIdx, :);
            YTrain = Y_num(trainIdx);
            XTest = X_all(testIdx, :);
            YTest = Y_num(testIdx);
            
        catch ME
            error('Veri bölme hatası: %s. featureMatrix ve labels boyutlarını kontrol edin.', ME.message);
        end
   
end
if ~isnumeric(YTrain)
    YTrainNum = double(ismember(string(YTrain), "PD"));
    YTestNum  = double(ismember(string(YTest), "PD"));
else
    if max(YTrain) <= 1
        YTrainNum = double(YTrain);
        YTestNum  = double(YTest);
    else
        YTrainNum = double(YTrain == 2); 
        YTestNum  = double(YTest == 2);
    end
end

numPD = sum(YTrainNum == 1);
numHC = sum(YTrainNum == 0);

combinedData = [XTrain, YTrainNum];
corrMatrix = corr(combinedData);
featNames = {'Ortalama (Mean)', 'Değişkenlik (CV)', 'Otokorelasyon (AC)', ...
             'Güç (Power)', 'Çarpıklık (Skew)', 'Basıklık (Kurt)', 'Hastalık Durumu'};

figure('Name', 'Öznitelik Korelasyon Matrisi', 'Color', 'w', 'NumberTitle', 'off');
h = heatmap(featNames, featNames, corrMatrix);

h.Title = 'Öznitelik ve Hastalık Korelasyonu';
h.Colormap = jet; 
h.ColorLimits = [-1 1]; 

[XTrainNorm, mu, sigma] = zscore(XTrain);
XTestNorm = (XTest - mu) ./ sigma;

save('trainedModel.mat', 'mu', 'sigma'); 

totalSamples = length(YTrainNum);
weightPD = totalSamples / (2 * numPD); 
weightHC = totalSamples / (2 * numHC); 

W = zeros(totalSamples, 1);
W(YTrainNum == 1) = weightPD;
W(YTrainNum == 0) = weightHC;

try
    Mdl = fitcsvm(XTrainNorm, YTrainNum, ...
        'KernelFunction', 'rbf', ...  
        'KernelScale', 'auto', ...    
        'Weights', W, ...             
        'Standardize', false);        
    Mdl = fitSVMPosterior(Mdl); 
catch
    warning('RBF Kernel zorlandı, Lineer Kernel deneniyor...');
    Mdl = fitcsvm(XTrainNorm, YTrainNum, 'KernelFunction', 'linear', 'Weights', W, 'Standardize', false);
    Mdl = fitSVMPosterior(Mdl);
end

[~, score] = predict(Mdl, XTestNorm);
scorePD = score(:,2); 

[FPR, TPR, T, AUC] = perfcurve(YTestNum, scorePD, 1);
[~, idx] = max(TPR - FPR); 
bestThreshold = T(idx);

YPred = double(scorePD >= bestThreshold);

C_mat = confusionmat(YTestNum, YPred);

if size(C_mat,1) > 1
    TN = C_mat(1,1); FP = C_mat(1,2);
    FN = C_mat(2,1); TP = C_mat(2,2);
    Sensitivity = TP / (TP + FN);
    Specificity = TN / (TN + FP);
    Accuracy = (TP + TN) / (TP + TN + FP + FN);
    F1_Score = 2*TP / (2*TP + FP + FN);
else
    Sensitivity = 0; Specificity = 0; Accuracy = 0; F1_Score = 0;
end

fprintf('Doğruluk (Accuracy) : %% %.2f\n', Accuracy * 100);
fprintf('Duyarlılık (Recall) : %% %.2f\n', Sensitivity * 100);
fprintf('Özgüllük (Spec)     : %% %.2f\n', Specificity * 100);
fprintf('F1-Skoru            : %% %.2f\n', F1_Score * 100);
fprintf('AUC Değeri          : %.4f\n', AUC);

save('trainedModel.mat', 'Mdl', 'mu', 'sigma', 'bestThreshold');

figure('Name', 'ROC Eğrisi', 'Color', 'w', 'NumberTitle', 'off');
plot(FPR, TPR, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'k--');
xlabel('Yanlış Pozitif Oranı (FPR)');
ylabel('Gerçek Pozitif Oranı (TPR)');
title(sprintf('ROC Eğrisi (AUC = %.2f)', AUC));
grid on;