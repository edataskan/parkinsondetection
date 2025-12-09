classdef ParkinsonApp < matlab.apps.AppBase

    properties (Access = public)
        UIFigure      matlab.ui.Figure
        Panel         matlab.ui.container.Panel
        LoadTxtButton matlab.ui.control.Button
        LoadVidButton matlab.ui.control.Button
        AnalyzeButton matlab.ui.control.Button
        StatusLabel   matlab.ui.control.Label
        Gauge         matlab.ui.control.LinearGauge
        ResultLabel   matlab.ui.control.Label
        
        UIAxesSignal  matlab.ui.control.UIAxes
        UIAxesVideo   matlab.ui.control.UIAxes
        UIAxesFeat    matlab.ui.control.UIAxes
       
        CurrentSignal
        Fs = 100; 
        IsVideoSource = false; 
    end

    methods (Access = private)

        function LoadVidButtonPushed(app, src, event)
            [file, path] = uigetfile({'*.mp4;*.avi;*.mov'}, 'Video Yükle (.mp4)');
            if isequal(file, 0), return; end
            
            videoPath = fullfile(path, file);
            app.StatusLabel.Text = 'Video İşleniyor (Lütfen Bekleyin)...';
            
            d = uiprogressdlg(app.UIFigure, 'Title', 'Video Analizi', ...
                'Message', 'Görüntü işleniyor ve yürüme sinyali çıkarılıyor...', 'Indeterminate', 'on');
            
            try
                vObj = VideoReader(videoPath);
                app.Fs = vObj.FrameRate;
                
                detector = vision.ForegroundDetector('NumTrainingFrames', 10, 'InitialVariance', 30*30);
                blob = vision.BlobAnalysis('BoundingBoxOutputPort', true, 'AreaOutputPort', true, ...
                    'CentroidOutputPort', true, 'MinimumBlobArea', 1000);
                
                extractedSignal = [];
                firstFrame = readFrame(vObj); 
                
                while hasFrame(vObj)
                    frame = readFrame(vObj);
                    mask = detector(frame);
                    mask = imopen(mask, strel('rectangle', [3,3]));
                    mask = imclose(mask, strel('rectangle', [15, 15]));
                    mask = imfill(mask, 'holes');
                    
                    [area, ~, centroid] = blob(mask);
                    
                    if ~isempty(area)
                        [~, idx] = max(area); 
                        y_pos = size(frame, 1) - centroid(idx, 2); 
                        extractedSignal = [extractedSignal; y_pos];
                    else
                        if ~isempty(extractedSignal)
                            extractedSignal = [extractedSignal; extractedSignal(end)];
                        end
                    end
                end
                
                if length(extractedSignal) < 50
                     error('Videoda yeterli hareket algılanamadı.');
                end

                app.CurrentSignal = smoothdata(extractedSignal, 'gaussian', 15);
                app.IsVideoSource = true;
                
                imshow(firstFrame, 'Parent', app.UIAxesVideo);
                title(app.UIAxesVideo, 'Video Kaynağı');
                
                plot(app.UIAxesSignal, app.CurrentSignal, 'b');
                title(app.UIAxesSignal, 'Videodan Çıkarılan Yürüme Ritmi');
                grid(app.UIAxesSignal, 'on');
                
                app.StatusLabel.Text = 'Video Sinyali Çıkarıldı.';
                app.AnalyzeButton.Enable = 'on';
                
            catch ME
                uialert(app.UIFigure, ['Video işleme hatası: ' ME.message], 'Hata');
            end
            close(d); 
        end

        function AnalyzeButtonPushed(app, src, event)
            if isempty(app.CurrentSignal), return; end
            
            app.StatusLabel.Text = 'Özellikler Hesaplanıyor...';
            
            signal = app.CurrentSignal;
            
            if app.IsVideoSource
                signal = smoothdata(signal, 'gaussian', 15); 
            else
                try
                    [b, a] = butter(4, [0.5/(app.Fs/2), 10/(app.Fs/2)], 'bandpass');
                    signal = filtfilt(b, a, signal);
                catch
                end
            end
            
            sig_std = std(signal);
            if sig_std == 0, sig_std = 1; end 
            sig_norm = (signal - mean(signal)) / sig_std;            
     
            try                
                max_peak_height = max(sig_norm);
                dynamic_threshold = max_peak_height * 0.4;
                if dynamic_threshold < 0.15
                    dynamic_threshold = 0.15;
                end                
                min_step_dist = floor(0.33 * app.Fs); 
                
                [~, locs] = findpeaks(sig_norm, ...
                     'MinPeakHeight', dynamic_threshold, ...        
                     'MinPeakDistance', min_step_dist, ...  
                     'MinPeakProminence', dynamic_threshold * 0.8); 

                if length(locs) < 3 && max_peak_height < 1.0
                     fprintf('Sinyal çok zayıf, hassas mod devreye giriyor...\n');
                     [~, locs] = findpeaks(sig_norm, ...
                          'MinPeakHeight', 0.1, ... 
                          'MinPeakDistance', floor(0.25 * app.Fs), ...
                          'MinPeakProminence', 0.05);
                end
           
            if length(locs) < 3
                uialert(app.UIFigure, 'Yeterli adım tespit edilemedi. Video çok karanlık veya hareket belirsiz.', 'Uyarı');
                app.ResultLabel.Text = 'Belirsiz';
                return;
            end
            
            stride_intervals = diff(locs) * (1/app.Fs);
           
            feat_1 = mean(stride_intervals);
            if feat_1 == 0, feat_2 = 0; 
            else 
                feat_2 = std(stride_intervals) / feat_1; 
            end
            if length(stride_intervals) > 1
                R = corrcoef(stride_intervals(1:end-1), stride_intervals(2:end));
                feat_3 = R(1, 2); if isnan(feat_3), feat_3 = 0; end
            else
                feat_3 = 0;
            end
            
            try
                if length(stride_intervals) >= 4 
                    [Pxx, F] = pwelch(stride_intervals - mean(stride_intervals), [], [], [], 100); 
                    low_freq_power = sum(Pxx(F < 0.5)); 
                    total_power = sum(Pxx);
                    if total_power == 0, feat_4 = 0; else, feat_4 = low_freq_power / total_power; end
                else
                    feat_4 = 0; 
                end
            catch
                feat_4 = 0; 
            end
            
            N = length(stride_intervals);
            mu_val = mean(stride_intervals);
            sigma_val = std(stride_intervals);
        
            if sigma_val == 0
                feat_5 = 0; feat_6 = 0;
            else
                standardized_data = (stride_intervals - mu_val) / sigma_val;
                feat_5 = (1/N) * sum(standardized_data.^3); 
                feat_6 = (1/N) * sum(standardized_data.^4); 
            end
            
            features = [feat_1, feat_2, feat_3, feat_4, feat_5, feat_6];
            
            try
                if exist('trainedModel.mat', 'file')
                    loaded = load('trainedModel.mat');
                    Mdl = loaded.Mdl;
                    
                    feat_norm = (features - loaded.mu) ./ loaded.sigma;
                    
                    [~, score] = predict(Mdl, feat_norm);
                    probPD = score(2);                     
                    app.Gauge.Value = probPD * 100;
                    
                    thresh = loaded.bestThreshold; 
                    
                    if probPD >= thresh
                        app.ResultLabel.Text = sprintf('RİSKLİ (PD)\n%%%.1f', probPD*100);
                        app.ResultLabel.FontColor = 'r';
                    else
                        app.ResultLabel.Text = sprintf('SAĞLIKLI (HC)\n%%%.1f', probPD*100);
                        app.ResultLabel.FontColor = [0 0.5 0]; 
                    end
                    
                    bar(app.UIAxesFeat, feat_norm);
                    title(app.UIAxesFeat, 'Özellik Dağılımı');
                    xticklabels(app.UIAxesFeat, {'Mean','CV','AC','Pow','Skew','Kurt'});
                    
                    hold(app.UIAxesSignal, 'off');
                    plot(app.UIAxesSignal, sig_norm, 'b');
                    hold(app.UIAxesSignal, 'on');
                    plot(app.UIAxesSignal, locs, sig_norm(locs), 'rv', 'MarkerFaceColor','r');
                    title(app.UIAxesSignal, sprintf('Tespit Edilen Adımlar: %d', length(locs)));
                    grid(app.UIAxesSignal, 'on');
                    
                    app.StatusLabel.Text = 'Analiz Tamamlandı.';
                else
                    uialert(app.UIFigure, 'trainedModel.mat bulunamadı! Önce one.m çalıştırın.', 'Hata');
                end
            catch ME
                uialert(app.UIFigure, ME.message, 'Hata');
            end
        end 

    end 
   
    methods (Access = public)
        function app = ParkinsonApp            
            app.UIFigure = uifigure('Name', 'Parkinson Teşhis', 'Position', [100 100 900 600]);
            app.Panel = uipanel(app.UIFigure, 'Title', 'Kontrol Paneli', 'Position', [20 20 220 560]);
            app.LoadVidButton = uibutton(app.Panel, 'push', 'Text', '🎥 Video Yükle (.mp4)', ...
                'Position', [20 450 180 35], 'BackgroundColor', [0.9 0.9 0.8], ...
                'ButtonPushedFcn', @app.LoadVidButtonPushed);
                
            app.AnalyzeButton = uibutton(app.Panel, 'push', 'Text', 'ANALİZ BAŞLAT', ...
                'Position', [20 380 180 45], 'Enable', 'off', 'BackgroundColor', [0 0.45 0.74], ...
                'FontColor', 'w', 'FontWeight', 'bold', 'ButtonPushedFcn', @app.AnalyzeButtonPushed);
            
            app.StatusLabel = uilabel(app.Panel, 'Position', [20 350 180 20], 'Text', 'Veri bekleniyor...');
            uilabel(app.Panel, 'Position', [20 280 180 20], 'Text', 'Hastalık Riski:', 'FontWeight', 'bold');
            app.Gauge = uigauge(app.Panel, 'linear', 'Position', [20 230 180 40], 'Limits', [0 100]);
            app.ResultLabel = uilabel(app.Panel, 'Position', [10 150 200 60], ...
                'Text', '---', 'HorizontalAlignment', 'center', 'FontSize', 20, 'FontWeight', 'bold');
            
            app.UIAxesVideo = uiaxes(app.UIFigure, 'Position', [260 320 300 250]);
            title(app.UIAxesVideo, 'Görüntü Kaynağı');
            app.UIAxesVideo.XTick = []; app.UIAxesVideo.YTick = [];
            
            app.UIAxesSignal = uiaxes(app.UIFigure, 'Position', [580 320 300 250]);
            title(app.UIAxesSignal, 'Zaman Serisi Sinyali');
            
            app.UIAxesFeat = uiaxes(app.UIFigure, 'Position', [260 20 620 250]);
            title(app.UIAxesFeat, 'Çıkarılan Özellikler (Normalize)');
        end
    end
end