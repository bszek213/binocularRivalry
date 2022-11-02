rootdir='/home/brianszekely/Desktop/ProjectsResearch/Binocular_Rivalry/Data/';
folders =dir(rootdir);
%% Stationary
len_people = length(3:length(folders));
R2_per_person_stand =zeros(len_people,1);
Hz_per_person_stand =zeros(len_people,1);
R2_per_person_walk =zeros(len_people,1);
Hz_per_person_walk =zeros(len_people,1);
z = 1;
randTimes = 5000;
fittingRange = [0.14 3];
for i = 3:length(folders)
    %Get data directory
    temp_loc = append(rootdir,string(folders(i).name),'/','Stationary','/','*.csv');
    files = dir(temp_loc);
    file1=fullfile(files.folder,[files.name]);
    %Read data in the matlab way
    data=readtable(file1);
    data=table2cell(data);
    %Per trial now
    Fitted_Fre = zeros(8,1);
    Fitted_Rsquare = zeros(8,1);
    for o = 1:8
        trial=find(cell2mat(data(:,4))==o);
        celldata=cell2mat(data(trial,11:13));
        celldata(:,3)=changem(celldata(:,3),-1,1);
        %Resp data to one column
        resp = zeros(length(celldata),1);
        for m = 1:length(celldata)
            if celldata(m,2) == 0 && celldata(m,3) == 0
                resp(m) = 0;
            elseif celldata(m,2) == 0 && celldata(m,3) == -1
                resp(m) = -1;
            elseif celldata(m,2) == 1 && celldata(m,3) == 0
                resp(m) = 1;
            else
                resp(m) = 0;
            end
        end
        %Fit frequencies
        Data = [celldata(:,1) resp];
        [Fitted_Fre(o),Fitted_Rsquare(o)] = SineWaveFitting(fittingRange,Data);
    end
    Hz_per_person_stand(z) = mean(Fitted_Fre);
    R2_per_person_stand(z) = mean(Fitted_Rsquare);
    z = z + 1;
end
%% Walking
len_people = length(3:length(folders));
R2_per_person =zeros(len_people,1);
Hz_per_person =zeros(len_people,1);
z = 1;
for i = 3:length(folders)
    %Get data directory
    temp_loc = append(rootdir,string(folders(i).name),'/','Walking','/',string(folders(i).name),'ResponsesWalking.csv');
    files = dir(temp_loc);
    file1=fullfile(files.folder,[files.name]);
    %Read data in the matlab way
    data=readtable(file1);
    data=table2cell(data);
    %Per trial now
    Fitted_Fre = zeros(8,1);
    Fitted_Rsquare = zeros(8,1);
    for o = 1:8
        trial=find(cell2mat(data(:,4))==o);
        celldata=cell2mat(data(trial,11:13));
        celldata(:,3)=changem(celldata(:,3),-1,1);
        %Resp data to one column
        resp = zeros(length(celldata),1);
        for m = 1:length(celldata)
            if celldata(m,2) == 0 && celldata(m,3) == 0
                resp(m) = 0;
            elseif celldata(m,2) == 0 && celldata(m,3) == -1
                resp(m) = -1;
            elseif celldata(m,2) == 1 && celldata(m,3) == 0
                resp(m) = 1;
            else
                resp(m) = 0;
            end
        end
        %Fit frequencies
        Data = [celldata(:,1) resp];
        [Fitted_Fre(o),Fitted_Rsquare(o)] = SineWaveFitting(fittingRange,Data);
%         parfor irand = 1:randTimes
%         %     rng;
%             Perm_order = randperm(size(Data,1));
%             Data_rand = [Data(:,1) Data(Perm_order,2)];
%             [~,Fitted_Rsquare_rand] = SineWaveFitting([Fitted_Fre Fitted_Fre],Data_rand)
%             rsquare_rand(irand) = Fitted_Rsquare_rand
% %             disp(irand)
%         end
        %PLOTTING EXAMPLE
        range_fit = linspace(fittingRange(1),fittingRange(2),50);
        Fitted_Fre_plot = zeros(length(range_fit),1);
        Fitted_Rsquare_plot = zeros(length(range_fit),1);
        test_fit = zeros(2,1);
        for p = 1:length(range_fit)
            test_fit = [range_fit(p) range_fit(p)];
            [Fitted_Fre_plot(p),Fitted_Rsquare_plot(p)] = SineWaveFitting(test_fit,Data);
        end
        plot(Fitted_Fre_plot,Fitted_Rsquare_plot)
        xlabel('Frequencies (Hz)')
        ylabel('R square')
    end
    %Save data per person
    Hz_per_person_walk(z) = mean(Fitted_Fre);
    R2_per_person_walk(z) = mean(Fitted_Rsquare);
    z = z + 1;
end