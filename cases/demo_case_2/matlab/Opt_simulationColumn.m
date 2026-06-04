   %Example of Distillation Column Multi-Objective Optimization using
%NSGA II Algorithm
%Tool Developed by: Andrés Felipe Abril - National University of Colombia

%% Aspen Plus - Matlab Link
global Aspen
    Aspen = actxserver('Apwn.Document.40.0');
    [stat,mess]=fileattrib;
    Aspen.invoke('InitFromArchive2',[mess.Name '\epsd.bkp']);
    Aspen.Visible = 1;
    Aspen.SuppressDialogs = 1;
    %Aspen.Engine.Run2(1);
    while Aspen.Engine.IsRunning == 1
        pause(0.5);
    end
%% NSGA II Algorithm Setup.
    xl = [4,  0.001, 0, 0,     4,  0.001, 0,  0,  4,  0.001, 0,     10,  10, 0.01, 0.01, 0.01];        %Lower bound vector of optimization variables.17
    xu = [200,  20,  1, 1,    200,   20,  1,  1,  100,  10,  1,  2000,  2000, 10,   10,    10];        %Upper bound vector of optimization variables.
    % Variable Description
       % x1: Stages Number of Distillation Column. Type: Integer.
       % x2: Reflux Ratio of Distillation Column. Type: Continuous.
       % x3: Normalized Distillation Feed stage number. Type: Continuous.
load("matlab.mat"); %Initial population. Optional.
NSGAparam.NumberObjM = 2;       % Number of Objective Functions. Min:2
NSGAparam.PopSize = 60;         % Population size
NSGAparam.Runs = 1;              % Number of runs
NSGAparam.MaxGen =1000;       % Max number of generations - stopping criteria
NSGAparam.FunObj = @Fun_Objective;   % Objective function and constraint evaluation
NSGAparam.VarSize = length(xl);
NSGAparam.LowerBound = xl;            % lower bound vector
NSGAparam.UpperBound = xu;            % upper bound vectorfor
NSGAparam.IntVar = [1];                % Vector that contains the position of the integer variable
NSGAparam.InitialPop = new_pop(:,1:16); %Inital population. Optional
NSGAparam.SaveResults = 'yes';        %Save Results of each generation
NSGAparam.PlotInterval = 1;          %Show Pareto Plot every 10 gen (plot interval)
NSGAparam.ConstNumber = 3;            % Number of constraints.
NSGAparam.CrossIndex = 20;            %Crossover parameter
NSGAparam.DistIndex = 20;               
NSGAparam.MutationProb = 0.5;         
NSGAparam.TruePF = [];               %Contains true pareto points for algorithm evaluation.                
Resultados = NSGA_II_Abril(NSGAparam);

    