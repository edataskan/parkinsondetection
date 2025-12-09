dataFolder = 'ParkinsonGaitAnalysis\data\raw'; 
fileList = dir(fullfile(dataFolder, '*.txt'));
Fs = 100;
gaitData = {}; 
rowCounter = 1; 

for i = 1:length(fileList)
    fileName = fileList(i).name;
    fullPath = fullfile(dataFolder, fileName);
    
    try
        data = dlmread(fullPath);
        if size(data, 2) >= 19
            signal = data(:, 18) + data(:, 19); % VGRF Toplamı
        else
            signal = data(:, 1);
        end
    catch
        continue; 
    end
    
    if ~isempty(strfind(upper(fileName), 'CO')), label = 'HC';
    elseif ~isempty(strfind(upper(fileName), 'PT')), label = 'PD';
    else, label = 'Unknown'; end
    
    if strcmp(label, 'HC')
        windowSize = 30 * Fs; 
        L = length(signal);
       
        numSegments = floor(L / windowSize);
        for k = 1:numSegments
             startIdx = (k-1)*windowSize + 1;
             endIdx = startIdx + windowSize - 1;
                
             newFileName = sprintf('%s_Part%d', fileName, k);
             gaitData{rowCounter, 1} = newFileName;
             gaitData{rowCounter, 2} = signal(startIdx:endIdx);
             gaitData{rowCounter, 3} = label;
             rowCounter = rowCounter + 1;
        end
        
    elseif strcmp(label, 'PD')
        gaitData{rowCounter, 1} = fileName;
        gaitData{rowCounter, 2} = signal;
        gaitData{rowCounter, 3} = label;
        rowCounter = rowCounter + 1;
    end
end

preprocessedGaitData = gaitData;
for i = 1:size(preprocessedGaitData, 1)
    raw_sig = preprocessedGaitData{i, 2};
    if isempty(raw_sig), continue; end
    
    raw_sig = raw_sig - mean(raw_sig); % DC ofset sil
    [~, locs] = findpeaks(raw_sig, 'MinPeakHeight', std(raw_sig)*1.5, 'MinPeakDistance', 40);
    
    if length(locs) > 3
        preprocessedGaitData{i, 2} = diff(locs) * (1/Fs); % Adım süreleri
    else
        preprocessedGaitData{i, 2} = [];
    end
end
