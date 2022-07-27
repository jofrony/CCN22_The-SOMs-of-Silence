% connection strengths %default values 

j_dt = 0.5;
j_max = 6;
j_idx = 0.2:j_dt:j_max;

je0=j_idx;
ji0=j_idx;
jei=j_idx;
jii=j_idx;
jee=j_idx;
jie=j_idx;


re=6;
ri=12;
tic 
count=0;
js=[];

for e=1:numel(je0)
    for i=1:numel(ji0)
        for ei=1:numel(jei)
            for ii=1:numel(jii)
                for ee=1:numel(jee)
                    for ie=1:numel(jie)
                        
                        if je0(e)/ji0(i)>jei(ei)/jii(ii) && jei(ei)/jii(ii)>jee(ee)/jie(ie)
                            
                            count=count+1;
                            js(count,:)=[je0(e),ji0(i),jei(ei),jii(ii),jee(ee),jie(ie)];
                        end
                        
                    end
                end
            end
        end
    end
end
toc


%%%%%%%%%%%%%%%%%



    re_g            = 6; %desired re 
    ri_g              = 12; %desired ri
    pei = 0.35;
    pie = 0.42;
    pii =0.39;
    pee = 0.06;

    gamma       = 1/4; %N factor, Ni/Ne
    beta_ei       = pei/pee; 
    beta_ie       = pie/pee;
    beta_ii         = pii/pee;
    
    R=[];
    for i=1:numel(js(:,1))
        je0=js(i,1);
        ji0=js(i,2);
        jei=js(i,3);
        jii=js(i,4);
        jee=js(i,5);
        jie=js(i,6);
    

    X=[je0;ji0];
    J=[jee,-(jei*gamma*beta_ei);(jie*beta_ie),-(jii*gamma*beta_ii)];
    
    R(i,:)=(-inv(J)*X)' * 100;
    
    end
    
    Re=[R(:,1)/re_g,R(:,2)/ri_g]
    
    figure
    scatter(Re(:,1),Re(:,2))
    xlim([0 2])
    ylim([0 2])
    
    Reabs=abs(1-Re);
    
    mError=sum(Reabs,2)
    [~,idx]=sort(mError);
    mError(idx,:);
    Re(idx,:);
    R(idx,:);
    js(idx,:);
    
    R;


Js_o = js(idx,:);
Js_t  = Js_o(1,:);
    


