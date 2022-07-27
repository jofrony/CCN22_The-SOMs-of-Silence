% Inputs:
% Ne number of e neurons
% Ni number of I neurons
% pee, pei, pie, pii; pab is the connection propability from b to a
%
%  [spktime_e,spkindex_e, spktime_i, spkindex_i, re, ri] = balanced_net( Ne, Ni, pee, pei, pie, pii, T, tau )
% 
% Outputs
%                   re                     mean firing rate of excitatory neurons
%                   ri                      mean firing rate of inhibitory neurons
%
% Optional inputs
%                   Ne                    {4136} number of excitatory neurons
%                   Ni                     {1034} number of inhibitory neurons
%                   pee                  {0.06} propability of connection between e to e 
%                   pei                   {0.35} propability of connection between i to e
%                   pie                   {0.42} propability of connection between e to i
%                   pii                    { 0.39} propability of connection between i to i
%                   T                      {1000} total time [ms]
%                   tau                  {10} the membrane timescale of the neuron [ms]
%  
%  Na   - number of neurons in a 
%  pab - propability of connection from b to a
%  Kab - number of expected connection from b to a
% 
% 26-Jul-22 HS

function [spktime_e,spkindex_e, spktime_i, spkindex_i, re, ri] = balanced_net( Ne, Ni, pee, pei, pie, pii, T, tau )

%--------------------------------------------------------------------%
% check inputs
%--------------------------------------------------------------------%

if nargin < 1 || isempty(Ne)
    Ne = 413;
%         Ne = 4136;

end

if nargin < 2 || isempty(Ni)
    Ni = 103;
%         Ni = 1034;

end

if nargin < 3 || isempty(pee)
    pee = 0.06;
end

if nargin < 4|| isempty(pei)
    pei = 0.35;
end

if nargin < 5|| isempty(pie)
    pie = 0.42;
end

if nargin < 6|| isempty(pii)
    pii =0.39;
end

if nargin < 7|| isempty(T)
    T =1000;
end

if nargin < 8|| isempty(tau)
    tau =10;
end

corr_binSize = 10; % correlation binSize [ms]

%external drives
je0=2.2;
ji0=0.6; 

% SOM
js0 = 0.15;
pse = 0;
% pse = 0.3;
pes = 0.23;
Ns   = round( ( Ne + Ni ) /10 );
jse=6.0; 
jes= 2;
Kse=pse*Ne; %expected # of connections (for e to e)'
Kes=pes*Ns; 




Kee=pee*Ne; %expected # of connections (for e to e)'
Kie=pie*Ne; 
Kei=pei*Ni; 
Kii=pii*Ni; 

% connection strengths %default values 
jee=2.5;
jie=6.0; 
jei=5.5;
jii=5.0; 
% e_rate_theory = ( je0 * jii - ji0 * jei ) / (jei * jie - jee*jii) * 10^3 / tau;
% i_rate_theory  = (je0 * jie - ji0 * jee) / (jei * jie - jee * jii) * 10^3 / tau;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

% scaling the true recurrent weights by 1/sqrt{K}
Jee=jee/sqrt(Kee); 
Jie=jie/sqrt(Kie); 
Jei=jei/sqrt(Kei); 
Jii=jii/sqrt(Kii); 
Jse = jse / sqrt(Kse);
Jes = jes / sqrt(Kes);

if Jee == inf
    Jee = 0;
end
if Jie == inf
    Jie = 0;
end
if Jei == inf
    Jei = 0;
end
if Jii == inf
    Jii = 0;
end
if Jse == inf
    Jse = 0;
end
if Jes == inf
    Jes = 0;
end

% scaling the feedforward weights by sqrt{K}
Je0=je0*sqrt(Kee); 
Ji0=ji0*sqrt(Kie); 
Js0 = js0 * sqrt(Kse);

if Js0 == 0
    Js0 = js0 * 10;
end

% membrane dynamics 
Vt=1;  %threshold
Vr=0;   %reset

%simulation details
dt                  = tau/500; %this should be accurate enough
maxspk         = T/dt; %pre-allocate space for the spike trains

%%%%%%%%connection matrices%%%%%%%

%E to E
cEE=zeros(Ne,Ne); %storing E to E connections
cEE_t=rand(Ne,Ne); %random matrix
cEE(cEE_t< pee)=1; %adjacency matrix (0 if no connection, 1 if connection)

%E to I
cIE=zeros(Ne,Ni);
cIE_t=rand(Ne,Ni);
cIE(cIE_t<pie)=1;

%I to E
cEI=zeros(Ni,Ne);
cEI_t=rand(Ni,Ne);
cEI(cEI_t<pei)=1;

%I to I
cII=zeros(Ni,Ni);
cII_t=rand(Ni,Ni);
cII(cII_t<pii)=1;

% E to S
cSE = zeros(Ne,Ns);
cSE_t=rand(Ne,Ns);
cSE(cSE_t<pse)=1;

% S to E
cES = zeros(Ns,Ne);
cES_t=rand(Ns,Ne);
cES(cES_t<pes)=1;


% intial conditions%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ve=rand(Ne,1);
Vi=rand(Ni,1);
Vs = rand(Ns,1);
Je0_vec=Je0.*ones(Ne,1);
Ji0_vec=Ji0.*ones(Ni,1);
Js0_vec = Js0 * ones(Ns,1);

% spiketime arrays; use a pre-allocated size
spktime_e=zeros(maxspk,1); % spike time array of E neurons
spkindex_e=zeros(maxspk,1); % identifity of the spike of each E neuron
spktime_i=zeros(maxspk,1); % spike time arrawy of I neuorns
spkindex_i=zeros(maxspk,1); % udentifity of the spikes of each I neuron 
spktime_s = zeros(maxspk,1);
spkindex_s = zeros(maxspk,1);

%intialize the spike time counter
counte=0;
counti=0;
counts=0;

% time loop%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for t= dt:dt:T 
     
    % zero the interactions from the last step
    kickee=zeros(Ne,1);
    kickei=zeros(Ne,1);
    kickie=zeros(Ni,1);
    kickii=zeros(Ni,1);

    kickse=zeros(Ns,1);
    kickes=zeros(Ne,1);


    % see which one of the E's spiked in the previous step
    index_spke=find(Ve>=Vt);  %find e spikers
    
    % if sombody spiked, update the spike times array and the spiker identifty
    if (~isempty(index_spke))
        spktime_e(counte+1:counte+length(index_spke))=t; %update spktime array
        spkindex_e(counte+1:counte+length(index_spke))=index_spke; %update spkindex array
        Ve(index_spke)=Vr;  %reset the spikers
        counte=counte+length(index_spke); %update the counter for the spike array
    end
    
    % see which one of the I's spiked in the previous step
    index_spki=find(Vi>=Vt); %find i spikers
    
    if (~isempty(index_spki))
        spktime_i(counti+1:counti+length(index_spki))=t;
        spkindex_i(counti+1:counti+length(index_spki))=index_spki;
        Vi(index_spki)=Vr;
        counti=counti+length(index_spki);
    end

  % see which one of the S spiked in the previous step
    index_spks=find(Vs>=Vt); %find i spikers
    
    if (~isempty(index_spks))
        spktime_s(counts+1:counts+length(index_spks))=t;
        spkindex_s(counts+1:counts+length(index_spks))=index_spks;
        Vs(index_spks)=Vr;
        counts=counts+length(index_spks);
    end
    
    %update e synaptic connectivity
    for j=1:length(index_spke) %loop over spikers
     
       kickee_index=find(cEE(index_spke(j),:)>0);  %find the post-synaptic targets of spiker j
       kickee(kickee_index)=kickee(kickee_index)+1; %update the post-synaptic targets kick
       
       kickie_index=find(cIE(index_spke(j),:)>0);
       kickie(kickie_index)=kickie(kickie_index)+1;
       
      kickse_index=find(cSE(index_spke(j),:)>0);
       kickse(kickse_index)=kickse(kickse_index)+1;
        
    end 
    
    %update i synaptic connectivity
    for j=1:length(index_spki)  
     
       kickei_index=find(cEI(index_spki(j),:)>0);
       kickei(kickei_index)=kickei(kickei_index)+1;
       
       kickii_index=find(cII(index_spki(j),:)>0);
       kickii(kickii_index)=kickii(kickii_index)+1;
        
    end 

        %update s synaptic connectivity
    for j=1:length(index_spks)  
       kickes_index=find(cES(index_spks(j),:)>0);
       kickes(kickes_index)=kickes(kickes_index)+1;
    end 

    
    % Find the change in Ve and Vi based on excitation and inhibition
    Ve = Ve+ (Jee*kickee ) - (Jei*kickei) - (Jes *kickes) ;  %kick the e neurons Vm 
%         Ve = Ve+ (Jee*kickee ) - (Jei*kickei) ;
    Vi = Vi+Jie*kickie-Jii*kickii;  %kick the i neurons Vm 
    Vs = Vs+Jse*kickse ;  %kick the s neurons Vm 

    %integrate the Vs with external inputs
    Ve=Ve+dt/tau*(-Ve+Je0_vec);  %e membrane integration 
    Vi=Vi+dt/tau*(-Vi+Ji0_vec);  %i membrane integration
    Vs=Vs+dt/tau*(-Vs+Js0_vec);  %s membrane integration

end 

non_zeroE = spktime_e ~= 0;
spktime_e = spktime_e(non_zeroE);
spkindex_e = spkindex_e(non_zeroE);

non_zeroI = spktime_i ~= 0;
spktime_i = spktime_i(non_zeroI);
spkindex_i = spkindex_i(non_zeroI);

non_zeroS = spktime_s~= 0;
spktime_s = spktime_s(non_zeroS);
spkindex_s = spkindex_s(non_zeroS);

re =counte/(T*Ne)*1000; % firing rate - the 1000 is to convert to Hz. 
ri = counti/(T*Ni)*1000;
rs = counts/(T*Ns)*1000;

figure;
subplot(3,1,1), plot(spktime_e,spkindex_e,'.k', 'MarkerSize',8); xlabel('Time (ms)', 'fontsize', 16, 'fontweight', 'b'); ylabel('E cell index', 'fontsize', 16, 'fontweight', 'b');
axis([0 T 0 Ne]);
title( sprintf( 'Ne=%d, re =%.2f Hz', Ne, re) );

subplot(3,1,2), plot(spktime_i,spkindex_i,'.k', 'MarkerSize',8); xlabel('Time (ms)', 'fontsize', 16, 'fontweight', 'b'); ylabel('I cell index', 'fontsize', 16, 'fontweight', 'b')
axis([0 T 0 Ni]);
title( sprintf( 'Ni=%d, ri =%.2f Hz', Ni, ri) );

subplot(3,1,3), plot(spktime_s,spkindex_s,'.k', 'MarkerSize',8); xlabel('Time (ms)', 'fontsize', 16, 'fontweight', 'b'); ylabel('S cell index', 'fontsize', 16, 'fontweight', 'b')
axis([0 T 0 Ns]);
title( sprintf( 'Ns=%d, rs =%.2f Hz', Ns, rs) );

suptitle( sprintf('pee=%.2f, pei=%.2f, pie=%.2f, pii=%.2f', pee, pei, pie, pii ) );

spike_train_corr(spktime_e, spkindex_e, corr_binSize);

return;
