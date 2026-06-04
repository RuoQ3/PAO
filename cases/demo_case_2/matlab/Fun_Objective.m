function [Obj, Cons] = Fun_Objective(X)
global Aspen
Stages1 =round(X(1));
RR1= X(2);
Feed_Stage1 = round((Stages1-6)*X(3) + 3);
Feed_Stage11 = round((Stages1-6)*X(4) + 3);
TowerDiameter1=Stages1-1;   

Stages2 =round(X(5));
RR2= X(6);
Feed_Stage2 = round((Stages2-6)*X(7) + 3);   
Feed_Stage22 = round((Stages2-6)*X(8) + 3);        
TowerDiameter2=Stages2-1; 

Stages3 =round(X(9));
RR3= X(10);
Feed_Stage3 = round((Stages3-6)*X(11) + 3);  
TowerDiameter3=Stages3-1;

SOL1=round(X(12));
SOL2=round(X(13));

P1= X(14);
P2= X(15);
P3= X(16);


Aspen.Tree.FindNode("\Data\Blocks\T1\Input\NSTAGE").Value = Stages1;
Aspen.Tree.FindNode("\Data\Blocks\T1\Input\BASIS_RR").Value =RR1;
Aspen.Tree.FindNode("\Data\Blocks\T1\Input\FEED_STAGE\F1").Value =Feed_Stage1; 
Aspen.Tree.FindNode("\Data\Blocks\T1\Input\FEED_STAGE\S1").Value =Feed_Stage11; 
Aspen.Tree.FindNode("\Data\Blocks\T1\Subobjects\Column Internals\INT-1\Input\CA_STAGE2\INT-1\CS-1").Value =TowerDiameter1;

Aspen.Tree.FindNode("\Data\Blocks\T2\Input\NSTAGE").Value = Stages2;
Aspen.Tree.FindNode("\Data\Blocks\T2\Input\BASIS_RR").Value = RR2;
Aspen.Tree.FindNode("\Data\Blocks\T2\Input\FEED_STAGE\F2").Value = Feed_Stage2;
Aspen.Tree.FindNode("\Data\Blocks\T2\Input\FEED_STAGE\S2").Value =Feed_Stage22; 
Aspen.Tree.FindNode("\Data\Blocks\T2\Subobjects\Column Internals\INT-1\Input\CA_STAGE2\INT-1\CS-1").Value =TowerDiameter2;

Aspen.Tree.FindNode("\Data\Blocks\T3\Input\NSTAGE").Value = Stages3;
Aspen.Tree.FindNode("\Data\Blocks\T3\Input\BASIS_RR").Value = RR3;
Aspen.Tree.FindNode("\Data\Blocks\T3\Input\FEED_STAGE\F3").Value = Feed_Stage3;
Aspen.Tree.FindNode("\Data\Blocks\T3\Subobjects\Column Internals\INT-1\Input\CA_STAGE2\INT-1\CS-1").Value =TowerDiameter3;

Aspen.Tree.FindNode("\Data\Streams\S1\Input\TOTFLOW\MIXED").Value =SOL1;
Aspen.Tree.FindNode("\Data\Streams\S2\Input\TOTFLOW\MIXED").Value =SOL2;

Aspen.Tree.FindNode("\Data\Blocks\T1\Input\PRES1").Value=P1;
Aspen.Tree.FindNode("\Data\Blocks\T2\Input\PRES1").Value=P2;
Aspen.Tree.FindNode("\Data\Blocks\T3\Input\PRES1").Value=P3;


    Aspen.Reinit; % Reinit simulation
    Aspen.Engine.Run2(1); %Run the simulation. (1) ---> Matlab isnt busy; (0) Matlab is Busy;
    time = 1;
    Error = 0;
    while Aspen.Engine.IsRunning == 1 % 1 --> If Aspen is running; 0 ---> If Aspen stop.
        pause(0.5);
        time = time+1;
        if time==40 % Control of simulation time.
            Aspen.Engine.Stop;
            Error = 1; 
        end
    end
Conv = Aspen.Tree.FindNode("\Data\Results Summary\Run-Status\Output\PER_ERROR").Value; %Convergence Assessment
    if Error == 0 && Conv == 0
        %Pur : ETOH Purity at distillation column top
        Pur1 =Aspen.Tree.FindNode("\Data\Streams\D1\Output\MOLEFRAC\MIXED\PROH").Value;
        Purity1 = 0.999 - Pur1; %This means: Pur > 0.999 mol frac.
        Pur2 =Aspen.Tree.FindNode("\Data\Streams\D2\Output\MOLEFRAC\MIXED\BUAC").Value;
        Purity2 = 0.999 - Pur2;
        Pur3 =Aspen.Tree.FindNode("\Data\Streams\D3\Output\MOLEFRAC\MIXED\WATER").Value;
        Purity3 = 0.999 - Pur3;
		
        %Recovery: ETOH Recovery at distillation column top.
        %Recovery > 0.99
        %Constraints Vector
        c(1,1) = Purity1;
        c(1,2) = Purity2;
        c(1,3) = Purity3;
        Cons = (c>0).*c; %Only accept unsatisfied constrainst: c > 0
        CAPEX = simulationCAPEX(); %Fixed Distillation Cost
        OPEX = simulationOPEX(); %Operating Distillation Cost
        Obj  = [CAPEX, OPEX]; %Objectives Function. CAPEX vs OPEX
    else
        Obj  = [2e30, 2e30]; %Penalty
        Cons = [1, 1, 1];
    end
end