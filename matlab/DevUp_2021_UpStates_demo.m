% Based on Rocha_2017
% ssaray@ucla.edu

close all
clear all


dt = 0.0001; %IN SECONDS
tmax   = 20/dt; %
nTrial = 1000; %1000
exp = 20; 
    
HOMEOSTATIC_FLAG = 1;
GRAPHICS = 1;
VIDEO = 1;
%savetrials = [1:10:nTrial,nTrial-9:nTrial-1]; 
savetrials= [1,199,995]; %Note that this numbers are rounded in the figure legend for simplicity

foldername=['cross-homeo_Up_Dev_demo'];

mkdir(foldername)


%% NETWORK PARAMETERS

F = @(x,gain,Thr) gain*max(0,x-Thr);

WEE = 3;    %3
WEI = 1;    %1 
WIE = 7;   %7
WII = 0.5;  %0.5

thetaE = 4.8;
thetaI = 25;
gainE = 1;
gainI = 4;

%W_MIN = 0.1;

Etau = 10/(dt*1000); %10
Itau = 2/(dt*1000);  %2

Beta = 0.4; %1 %0.5
tauA = 500;


%Evoked
EvokedOn = 0.250/dt;
EvokedDur = 0.01/dt;
EvokedAmp = 0;


hR = zeros(tmax,2); %history of Inh and Ex rate and adaptation


E_MAX = 100; %100
I_MAX = 250; %250
min_ExAvg = 1;
min_InhAvg =1;


%% PLASTICITY STUFF


ExSet = 5;
InhSet = 14;

tau_trial = 10; 
UpThreshold = 2; 


mean_FCaUp =0;


W_MIN= 0.1;


alpha=0.0001;



rng(42) %fixed seed

tic

%% GRAPHICS

if GRAPHICS
h=figure('Renderer', 'painters', 'Position', [1 1 1200 900]);  
set(gcf,'color','w');
subplot(3,1,1)
colormap = [ 255/255,51/255,51/255; 0 0.5 0; 0 0 1];
set(gca,'colororder',colormap);
hold on
plot(dt:dt:tmax*dt,hR,'ydatasource','hR','linewidth',3);
line([dt tmax*dt],[ExSet ExSet],'color',[0 0.5 0],'linestyle','--','linewidth',2);
line([dt tmax*dt],[InhSet InhSet],'color',[255/255,51/255,51/255],'linestyle','--','linewidth',2);
ylabel('E/I (Hz)')
xlabel('Time (sec)')
set(gca,'FontSize',20)
set(gca,'linew',2)
set(gca, 'box', 'off')
ylim([0 35])


subplot(3,1,2)
plot(zeros(nTrial,1),'color',[255/255,51/255,51/255],'ydatasource','trialhistFCaInh','linewidth',3);
hold on
plot(zeros(nTrial,1),'color',[0 0.5 0],'ydatasource','trialhistFCa','linewidth',3);
line([1 nTrial],[ExSet ExSet],'color',[0 0.5 0],'linestyle','--','linewidth',2);
line([1 nTrial],[InhSet InhSet],'color',[255/255,51/255,51/255],'linestyle','--','linewidth',2);
ylabel('Mean E/I (Hz)')
set(gca,'FontSize',20)
set(gca,'linew',2)
set(gca, 'box', 'off')
ylim([0 20])


subplot(3,1,3)
plot(zeros(nTrial,1),'color',[18/255, 181/255, 143/255],'ydatasource','trialhistWEE','linewidth',3);
hold on
plot(zeros(nTrial,1),'color',[18/255, 181/255, 143/255],'ydatasource','trialhistWEI','linestyle',':','linewidth',3);
plot(zeros(nTrial,1),'color',[217/255, 68/255, 220/255],'ydatasource','trialhistWIE','linewidth',3);
plot(zeros(nTrial,1),'color',[217/255, 68/255, 220/255],'ydatasource','trialhistWII','linestyle',':','linewidth',3);
xlim([0 nTrial])
ylabel('Weights')
xlabel('Trials')
set(gca,'FontSize',20)
set(gca,'linew',2)
set(gca, 'box', 'off')
legend('WEE','WEI','WIE','WII','LineWidth',1)
ylim([0 8])

end

%%

    
 %% Trial history variables
trialhistmean_FCaUp = NaN(nTrial,1);
trialhistFCa      = NaN(nTrial,1);
trialhistFCaInh   = NaN(nTrial,1);
trialhistWEE      = NaN(nTrial,1);
trialhistWEI      = NaN(nTrial,1);
trialhistWIE      = NaN(nTrial,1);
trialhistWII      = NaN(nTrial,1);  
trialhistUpDur    = NaN(nTrial,1); 
trialhistUpFreq    = NaN(nTrial,1); 

%% Init Variables

WInit = [WEE WEI WIE WII];
   
ExAvg = 0;
InhAvg = 0;

      
% Ornstein�Uhlenbeck Noise
OUtau = 1/10;
OUmu = 0;
OUsigma = 1.2; %1.1; %in the paper the sigma is 3.5??? sqrt(3.5) = 1.87 | may be mean sigma is the variance?
OUE = 0;
OUI = 0;


hR = zeros(tmax,2); %history of Inh and Ex rate and adaptation

counter = 0;
   
   for trial=1:nTrial
      
      
      E = 0;
      I = 0;
      a = 0;
      
      
      fCa = zeros(1,tmax);  %instantaneous fast Ca sensor
      fCaInh = zeros(1,tmax);  %instantaneous fast Ca sensor
      totUpTime = 0;
      totUpTimeI = 0;

      
      hR = zeros(tmax,2); %history of Inh and Ex rate and adaptation
      
      for t=1:tmax
         
         
         
         OUE = OUE + OUtau*(OUmu-OUE) + OUsigma*randn; %Ornstein-Uhlenbeck Noise for excitatory unit
         OUI = OUI + OUtau*(OUmu-OUI) + OUsigma*randn; %Ornstein-Uhlenbeck Noise for inhibitory unit
         
         E = E + (-E + F(WEE*E - WEI*I - a + OUE,gainE,thetaE) )/Etau;
         I = I + (-I + F(WIE*E - WII*I + OUI,gainI,thetaI) )/Itau;
         
         a = a + (-a + Beta*E)/tauA; %ADAPTATION 

         if E>E_MAX; E = E_MAX; end
         if I>I_MAX; I = I_MAX; end
                  
         % Ca Sensors
         
         fCa(:,t) = E;         
         fCaInh(:,t) = I;
     

         hR(t,:) = [I E];
         
      end
      
       [UpFreq, UpDur] = RasterFindUpStates(hR(:,2)',1,0.1);

      %% HOMEOSTASIS
         
         fCa_Up = fCa(fCa>UpThreshold);
         fCaInh_Up = fCaInh(fCaInh>UpThreshold);
         
         if isempty(fCa_Up)
             fCa_Up = 0;
         end
         if isempty(fCaInh_Up)
             fCaInh_Up = 0;
         end
         
         mean_FCaUp=mean(fCa_Up,2);
         ExAvg    = ExAvg  + (-ExAvg  + mean_FCaUp)/tau_trial;
         InhAvg   = InhAvg + (-InhAvg + mean(fCaInh_Up,2))/tau_trial;
         

      
      if HOMEOSTATIC_FLAG
          

          EAvg =  max(1,ExAvg); %Presynaptic factor on the rules is rectified with a minimum value. 
          IAvg = max(1,InhAvg); 
          
            newWEE = WEE + alpha*EAvg*(InhSet-IAvg);
            newWEI = WEI - alpha*IAvg*(InhSet-IAvg);
            newWIE = WIE - alpha*EAvg*(ExSet-EAvg); 
            newWII = WII + alpha*IAvg*(ExSet-EAvg); 
                     
         
         WEE = newWEE; WEI = newWEI; WIE = newWIE; WII = newWII;
         
         if WEE<W_MIN; WEE = W_MIN; end
         if WEI<W_MIN; WEI = W_MIN; end
         if WIE<W_MIN; WIE = W_MIN; end
         if WII<W_MIN; WII = W_MIN; end
      
      end
      
      
      trialhistmean_FCaUp(trial) = mean_FCaUp;
      trialhistFCa(trial) = ExAvg;
      trialhistFCaInh(trial) = InhAvg;
      trialhistWEE(trial) = WEE;
      trialhistWEI(trial) = WEI;
      trialhistWIE(trial) = WIE;
      trialhistWII(trial) = WII;
      trialhistUpDur(trial) = mean(UpDur);
      trialhistUpFreq(trial) = UpFreq;


%%
      if GRAPHICS
     
      refreshdata %refresh graphics
      drawnow
      
        if ismember(trial,savetrials)
            saveas(gcf,[foldername,'\Ex',num2str(ExSet),'Inh',num2str(InhSet),'Trial',num2str(trial)],'epsc') 
        end
      end
%%
      if VIDEO
      counter = counter+1;
      frames(counter) = getframe(h); 
      end
      
   end
   
   
   
  %% 
   
   if VIDEO
    video = VideoWriter([foldername,'/exp',num2str(exp)]);
    video.FrameRate = 5;
    open(video)
    writeVideo(video,frames);
    close(video)
    clear video
    clear frames
   end
   
   
   close all

save([foldername,'/temp','Ex',num2str(ExSet),'Inh',num2str(InhSet)])
toc