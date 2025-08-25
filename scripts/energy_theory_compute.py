import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

T0 = 81
H0, W0 = 480, 832
N = 32
S0 = 50
m_text = 512
d = 2048
f_mlp = 4
vae_stride_t = 4
vae_stride_s = 8
patch_h, patch_w = 2, 2
g = 2
d_tau = 256
Z_DIM = 16
BASE = 96
USE_EXACT_VAE = True
K_enc = 2.0e9
K_dec = 2.0e9
theta_peak = 989e12
mu_eff = 0.456
P_max = 700.0
latency_overhead = 0.0
energy_overhead = 0.0
S_values = np.arange(1, 200, 4)
scale_values = np.linspace(0.5, 2.0, 40)
T_values = np.arange(4, 100, 2)

COMP_ORDER = ["text", "temb", "self", "cross", "mlp", "dec"]
COMP_LABELS = {
    "text": "Text encoder",
    "temb": "Timestep MLP",
    "self": "Self-attention DiT",
    "cross": "Cross-attention DiT",
    "mlp": "MLP DiT",
    "dec": "VAE decoder",
}
COMP_COLORS = {
    "text": "#E15759",
    "temb": "#76B7B2",
    "self": "#F58518",
    "cross": "#54A24B",
    "mlp": "#B279A2",
    "dec": "#93E8CD",
}

def conv3d_flops(cin, cout, kt, kh, kw, T, H, W):
    return 2 * kt * kh * kw * cin * cout * T * H * W

def conv2d_flops(cin, cout, kh, kw, T, H, W):
    return 2 * kh * kw * cin * cout * T * H * W

def attn2d_middle_flops(C, T, H, W):
    L = H * W
    toqkv = 2 * C * (3*C) * L
    proj  = 2 * C * C * L
    mat   = 4 * (L * L) * C
    return T * (toqkv + proj + mat)

def wan_vae_flops_exact(T0, H0, W0, z=Z_DIM, d0=BASE):
    T0, H0, W0 = int(T0), int(H0), int(W0)
    T1, H1, W1 = T0, math.ceil(H0/2), math.ceil(W0/2)
    T2, H2, W2 = math.ceil(T1/2), math.ceil(H1/2), math.ceil(W1/2)
    T3, H3, W3 = math.ceil(T2/2), math.ceil(H2/2), math.ceil(W2/2)
    Fenc = conv3d_flops(3, d0, 3,3,3, T0,H0,W0)
    Fenc += 4 * conv3d_flops(d0, d0, 3,3,3, T0,H0,W0)
    Fenc += conv2d_flops(d0, d0, 3,3, T0, H1, W1)
    Fenc += conv3d_flops(96,192,3,3,3, T1,H1,W1) + conv3d_flops(192,192,3,3,3, T1,H1,W1) + conv3d_flops(96,192,1,1,1, T1,H1,W1)
    Fenc += 2 * conv3d_flops(192,192,3,3,3, T1,H1,W1)
    Fenc += conv2d_flops(192,192,3,3, T1, H2, W2) + conv3d_flops(192,192,3,1,1, T2, H2, W2)
    Fenc += conv3d_flops(192,384,3,3,3, T2,H2,W2) + conv3d_flops(384,384,3,3,3, T2,H2,W2) + conv3d_flops(192,384,1,1,1, T2,H2,W2)
    Fenc += 2 * conv3d_flops(384,384,3,3,3, T2,H2,W2)
    Fenc += conv2d_flops(384,384,3,3, T2, H3, W3) + conv3d_flops(384,384,3,1,1, T3, H3, W3)
    Fenc += 4 * conv3d_flops(384,384,3,3,3, T3,H3,W3)
    Fenc += 4 * conv3d_flops(384,384,3,3,3, T3,H3,W3) + attn2d_middle_flops(384, T3, H3, W3)
    Fenc += conv3d_flops(384, 2*z, 3,3,3, T3,H3,W3)
    Fdec = conv3d_flops(z,384,3,3,3, T3,H3,W3)
    Fdec += 4 * conv3d_flops(384,384,3,3,3, T3,H3,W3) + attn2d_middle_flops(384, T3,H3,W3)
    Fdec += 6 * conv3d_flops(384,384,3,3,3, T3,H3,W3)
    Fdec += conv3d_flops(384,768,3,1,1, 2*T3,H3,W3)
    Fdec += conv2d_flops(384,192,3,3, 2*T3,2*H3,2*W3)
    T35,H2,W2=2*T3,2*H3,2*W3
    Fdec += conv3d_flops(192,384,3,3,3, T35,H2,W2)+conv3d_flops(384,384,3,3,3, T35,H2,W2)+conv3d_flops(192,384,1,1,1, T35,H2,W2)
    Fdec += 2*conv3d_flops(384,384,3,3,3, T35,H2,W2)
    Fdec += conv3d_flops(384,768,3,1,1, 4*T3,H2,W2)
    Fdec += conv2d_flops(384,192,3,3, 4*T3,4*H3,4*W3)
    T2,H1,W1=4*T3,4*H3,4*W3
    Fdec += 6*conv3d_flops(192,192,3,3,3,T2,H1,W1)
    Fdec += conv2d_flops(192,96,3,3, T2,8*H3,8*W3)
    Fdec += conv3d_flops(96,3,3,3,3, T2,8*H3,8*W3)
    return Fenc, Fdec

def flops_vae_enc(T,H,W):
    if USE_EXACT_VAE: return wan_vae_flops_exact(T,H,W,Z_DIM,BASE)[0]
    return K_enc*T*H*W

def flops_vae_dec(T,H,W):
    if USE_EXACT_VAE: return wan_vae_flops_exact(T,H,W,Z_DIM,BASE)[1]
    return K_dec*T*H*W

def flops_t5_layer(m_text,d_text,f_ff=4):
    F_attn=8.0*m_text*(d_text**2)+4.0*(m_text**2)*d_text
    F_ffn=4.0*f_ff*m_text*(d_text**2)
    return F_attn+F_ffn

def flops_text_encoder_total(m_text,d_text,L_text,f_ff=4,passes=2):
    return passes*L_text*flops_t5_layer(m_text,d_text,f_ff)

def flops_timestep_mlp_per_layer(d_model,d_tau=d_tau):
    return 2*d_tau*d_model+14*d_model**2

def flops_dit_per_layer(T,H,W,d,f_mlp=4,m_text=512,vae_t=4,vae_s=8,p_h=2,p_w=2):
    ell=(1.0+T/float(vae_t))*(H/(vae_s*p_h))*(W/(vae_s*p_w))
    F_self=8.0*ell*d**2+4.0*(ell**2)*d
    F_cross_step=4.0*ell*d**2+4.0*m_text*d**2+4.0*ell*m_text*d
    F_mlp=4.0*f_mlp*ell*d**2
    return F_self,F_cross_step,F_mlp

def flops_total_breakdown(T,H,W,S,N,d,f_mlp=4,m_text=512,
                          enc=False, include_text=True,
                          d_text=1024, L_text=24, f_text=4, text_passes=2,
                          include_timestep=True, temb_layers=2, temb_hidden=2,
                          g=g):
    F_enc = flops_vae_enc(T,H,W) if enc else 0.0
    F_dec = flops_vae_dec(T,H,W)

    F_self_l, F_cross_step_l, F_mlp_l = flops_dit_per_layer(T,H,W,d,f_mlp,m_text)
    F_self  = S * N * g * F_self_l
    F_cross = S * N * g * F_cross_step_l
    F_mlp   = S * N * g * F_mlp_l

    F_text = flops_text_encoder_total(m_text,d_text,L_text,f_text,text_passes) if include_text else 0.0
    F_temb = (S*N*flops_timestep_mlp_per_layer(d_model=d,d_tau=d_tau)) if include_timestep else 0.0

    F_total = F_enc + F_self + F_cross + F_mlp + F_dec + F_text + F_temb

    res = {}
    if enc:
        res["enc"] = F_enc
    if include_text:
        res["text"] = F_text
    if include_timestep:
        res["temb"] = F_temb

    res["self"]  = F_self
    res["cross"] = F_cross
    res["mlp"]   = F_mlp
    res["dec"]   = F_dec
    res["total"] = F_total

    return res


def latency_energy_from_flops(F_total,theta_peak,mu_eff,P_max,lat_over=0.0,en_over=0.0):
    T_s=F_total/(mu_eff*theta_peak)+lat_over
    E_j=P_max*T_s+en_over
    return T_s,E_j

def build_curves_vary_S(S_values,T,H,W,enc=False,include_text=True,include_timestep=True,d_text=1024,L_text=24,f_text=4,text_passes=2,temb_layers=2,temb_hidden=None):
    parts=[]
    if enc: parts.append("enc")
    if include_text: parts.append("text")
    if include_timestep: parts.append("temb")
    parts+=["self","cross","mlp","dec"]
    E_comp,T_comp={p:[] for p in parts},{p:[] for p in parts}
    E_total,T_total=[],[]
    for S in S_values:
        fb=flops_total_breakdown(T,H,W,S,N,d,f_mlp,m_text,enc,include_text,d_text,L_text,f_text,text_passes,include_timestep,temb_layers,temb_hidden)
        for p in parts:
            T_s,E_j=latency_energy_from_flops(fb[p],theta_peak,mu_eff,P_max,latency_overhead,energy_overhead)
            T_comp[p].append(T_s); E_comp[p].append(E_j)
        T_s_tot,E_j_tot=latency_energy_from_flops(fb["total"],theta_peak,mu_eff,P_max,latency_overhead,energy_overhead)
        T_total.append(T_s_tot); E_total.append(E_j_tot)
    return np.array(T_total),np.array(E_total),{p:np.array(T_comp[p]) for p in parts},{p:np.array(E_comp[p]) for p in parts}

def build_curves_vary_scale(scale_values,T,H0,W0,S,enc=False,include_text=True,include_timestep=True,d_text=1024,L_text=24,f_text=4,text_passes=2,temb_layers=2,temb_hidden=None):
    parts=[]
    if enc: parts.append("enc")
    if include_text: parts.append("text")
    if include_timestep: parts.append("temb")
    parts+=["self","cross","mlp","dec"]
    E_comp,T_comp={p:[] for p in parts},{p:[] for p in parts}
    E_total,T_total=[],[]
    for s in scale_values:
        H=int(round(H0*s)); W=int(round(W0*s))
        fb=flops_total_breakdown(T,H,W,S,N,d,f_mlp,m_text,enc,include_text,d_text,L_text,f_text,text_passes,include_timestep,temb_layers,temb_hidden)
        for p in parts:
            T_s,E_j=latency_energy_from_flops(fb[p],theta_peak,mu_eff,P_max,latency_overhead,energy_overhead)
            T_comp[p].append(T_s); E_comp[p].append(E_j)
        T_s_tot,E_j_tot=latency_energy_from_flops(fb["total"],theta_peak,mu_eff,P_max,latency_overhead,energy_overhead)
        T_total.append(T_s_tot); E_total.append(E_j_tot)
    return np.array(T_total),np.array(E_total),{p:np.array(T_comp[p]) for p in parts},{p:np.array(E_comp[p]) for p in parts}

def build_curves_vary_T(T_values,H,W,S,enc=False,include_text=True,include_timestep=True,d_text=1024,L_text=24,f_text=4,text_passes=2,temb_layers=2,temb_hidden=None):
    parts=[]
    if enc: parts.append("enc")
    if include_text: parts.append("text")
    if include_timestep: parts.append("temb")
    parts+=["self","cross","mlp","dec"]
    E_comp,T_comp={p:[] for p in parts},{p:[] for p in parts}
    E_total,T_total=[],[]
    for T in T_values:
        fb=flops_total_breakdown(T,H,W,S,N,d,f_mlp,m_text,enc,include_text,d_text,L_text,f_text,text_passes,include_timestep,temb_layers,temb_hidden)
        for p in parts:
            T_s,E_j=latency_energy_from_flops(fb[p],theta_peak,mu_eff,P_max,latency_overhead,energy_overhead)
            T_comp[p].append(T_s); E_comp[p].append(E_j)
        T_s_tot,E_j_tot=latency_energy_from_flops(fb["total"],theta_peak,mu_eff,P_max,latency_overhead,energy_overhead)
        T_total.append(T_s_tot); E_total.append(E_j_tot)
    return np.array(T_total),np.array(E_total),{p:np.array(T_comp[p]) for p in parts},{p:np.array(E_comp[p]) for p in parts}

def stacked_area(ax,x,comp_dict,title,xlabel,ylabel,enc=False,include_text=True,include_timestep=True):
    order=[]
    if enc: order.append("enc")
    if include_text: order.append("text")
    if include_timestep: order.append("temb")
    order+=["self","cross","mlp","dec"]
    labels={}
    if enc: labels["enc"]="VAE encoder"
    if include_text: labels["text"]="Text encoder"
    if include_timestep: labels["temb"]="Timestep MLP"
    labels={**labels,"self":"Self-attention blocks DiT","cross":"Cross-attention blocks DiT","mlp":"MLP blocks DiT","dec":"VAE decoder"}
    y=[comp_dict[k] for k in order]
    ax.stackplot(x,y,labels=[labels[k] for k in order],alpha=0.8)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True,alpha=0.3); ax.legend(loc="upper left",frameon=False)

def theory_components_wh_s(T,H,W,S):
    fb=flops_total_breakdown(T,H,W,S,N,d,f_mlp,m_text)
    e_wh,t_s={},{}
    for k in COMP_ORDER:
        t,e=latency_energy_from_flops(fb[k],theta_peak,mu_eff,P_max,latency_overhead,energy_overhead)
        e_wh[k]=e/3600.0
        t_s[k]=t
    return e_wh,t_s,sum(e_wh.values()),sum(t_s.values())

def build_series(df,x_col,w_col,h_col,t_col,s_col,obs_energy_scale,obs_energy_col,obs_time_col):
    xs=df[x_col].to_numpy()
    e_comp={k:[] for k in COMP_ORDER}
    t_comp={k:[] for k in COMP_ORDER}
    e_tot,t_tot=[],[]
    for _,r in df.iterrows():
        e_wh_dict,t_s_dict,e_total,t_total=theory_components_wh_s(int(r[t_col]),int(r[h_col]),int(r[w_col]),int(r[s_col]))
        for k in COMP_ORDER:
            e_comp[k].append(e_wh_dict[k])
            t_comp[k].append(t_s_dict[k])
        e_tot.append(e_total)
        t_tot.append(t_total)
    for k in COMP_ORDER:
        e_comp[k]=np.array(e_comp[k])
        t_comp[k]=np.array(t_comp[k])
    e_tot=np.array(e_tot)
    t_tot=np.array(t_tot)
    e_obs=df[obs_energy_col].to_numpy()*obs_energy_scale
    t_obs=df[obs_time_col].to_numpy()
    return xs,e_comp,t_comp,e_obs,t_obs,e_tot,t_tot

def stacked_theory_with_practice(ax,x,comp_dict,obs,title,xlabel,ylabel,show_total_outline=True):
    ys=[comp_dict[k] for k in COMP_ORDER]
    labels=[COMP_LABELS[k] for k in COMP_ORDER]
    colors=[COMP_COLORS[k] for k in COMP_ORDER]
    ax.stackplot(x,ys,labels=labels,colors=colors,alpha=0.8,linewidth=0.5)
    if show_total_outline:
        total_theory=np.sum(np.vstack(ys),axis=0)
        ax.plot(x,total_theory,color="black",linestyle="--",linewidth=1.5,label="Theory (total)")
    ax.plot(x,obs,color="#4d4d4d",marker="o",linestyle="-",linewidth=1.5,label="Practice",alpha=0.8)
    ax.set_title(title,fontsize=20)
    ax.set_xlabel(xlabel,fontsize=16)
    ax.set_ylabel(ylabel,fontsize=16)
    ax.tick_params(axis='both',which='major',labelsize=12)
    ax.legend(loc="upper left",frameon=False,fontsize=12)
    ax.grid(True,alpha=0.3)

def mean_percentage_error(y_true,y_pred):
    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)
    return float(np.mean(np.abs((y_true-y_pred)/y_true))*100.0)

def estimate_mu_and_overhead_from_dfs(dfs,theta_peak,build_args,time_col):
    F_list,T_list=[],[]
    for df in dfs:
        for _,r in df.iterrows():
            fb=flops_total_breakdown(int(r['num_frames']),int(r['height']),int(r['width']),int(r['steps']),build_args['N'],build_args['d'],build_args['f_mlp'],build_args['m_text'])
            F_list.append(fb['total'])
            T_list.append(float(r[time_col]))
    F=np.asarray(F_list)
    T=np.asarray(T_list)
    X=np.column_stack([F,np.ones_like(F)])
    coeff,_,_,_=np.linalg.lstsq(X,T,rcond=None)
    a_hat,b_hat=coeff[0],coeff[1]
    mu_hat=float(np.clip(1.0/(a_hat*theta_peak),1e-3,1.0))
    T_pred=a_hat*F+b_hat
    ss_res=np.sum((T-T_pred)**2)
    ss_tot=np.sum((T-np.mean(T))**2)
    R2=float(1.0-ss_res/ss_tot) if ss_tot>0 else float("nan")
    return mu_hat,float(b_hat),R2

def ensure_outdir(p):
    os.makedirs(p,exist_ok=True)

def concat_all_splits(ds):
    frames=[]
    for split in ds.keys():
        frames.append(ds[split].to_pandas())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames,ignore_index=True)

def plot_models_bars(df,outdir,dpi,energy_col,time_col,model_col):
    models=list(df[model_col].astype(str).values)
    e=df[energy_col].values
    t=df[time_col].values
    x=np.arange(len(models))
    plt.figure(figsize=(10,5))
    ax=plt.gca()
    ax.bar(x,e,width=0.6,edgecolor="black",alpha=0.9)
    ax.set_xticks(x,models,rotation=45,ha="right")
    ax.set_ylabel("Energy (Wh)")
    ax.set_title("Observed GPU Energy by Model")
    ax.grid(True,axis="y",alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"models_energy_wh.png"),dpi=dpi)
    plt.figure(figsize=(10,5))
    ax=plt.gca()
    ax.bar(x,t,width=0.6,edgecolor="black",alpha=0.9)
    ax.set_xticks(x,models,rotation=45,ha="right")
    ax.set_ylabel("Duration (s)")
    ax.set_title("Observed Generation Time by Model")
    ax.grid(True,axis="y",alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"models_time_s.png"),dpi=dpi)

def main():
    global USE_EXACT_VAE, theta_peak, mu_eff, P_max, latency_overhead, energy_overhead
    parser=argparse.ArgumentParser(prog="t2v-latency-energy")
    parser.add_argument("--outdir",type=str,default="./../results")
    parser.add_argument("--theta-peak",type=float,default=theta_peak)
    parser.add_argument("--mu",type=float,default=mu_eff)
    parser.add_argument("--p-max",type=float,default=P_max)
    parser.add_argument("--latency-overhead",type=float,default=latency_overhead)
    parser.add_argument("--energy-overhead",type=float,default=energy_overhead)
    parser.add_argument("--use-exact-vae",action="store_true")
    parser.add_argument("--obs-energy-scale",type=float,default=1000.0)
    parser.add_argument("--wan-ds",type=str,default="VideoKilledEnergyBudget/text2video-wan-energy-benchmark")
    parser.add_argument("--wan-vary-kind-col",type=str,default="vary_kind")
    parser.add_argument("--wan-kind-frames",type=str,default="frames")
    parser.add_argument("--wan-kind-steps",type=str,default="steps")
    parser.add_argument("--wan-kind-res",type=str,default="res")
    parser.add_argument("--energy-col",type=str,default="energy_generate_gpu")
    parser.add_argument("--time-col",type=str,default="duration_generate")
    parser.add_argument("--width-col",type=str,default="width")
    parser.add_argument("--height-col",type=str,default="height")
    parser.add_argument("--frames-col",type=str,default="num_frames")
    parser.add_argument("--steps-col",type=str,default="steps")
    parser.add_argument("--models-ds",type=str,default="VideoKilledEnergyBudget/text2video-energy-benchmark")
    parser.add_argument("--models-model-col",type=str,default="model_name")
    parser.add_argument("--dpi",type=int,default=300)
    args=parser.parse_args()

    ensure_outdir(args.outdir)
    USE_EXACT_VAE=args.use_exact_vae
    theta_peak=args.theta_peak
    mu_eff=args.mu
    P_max=args.p_max
    latency_overhead=args.latency_overhead
    energy_overhead=args.energy_overhead

    T_tot_S,E_tot_S,T_comp_S,E_comp_S=build_curves_vary_S(S_values,T0,H0,W0)
    T_tot_sc,E_tot_sc,T_comp_sc,E_comp_sc=build_curves_vary_scale(scale_values,T0,H0,W0,S0)
    T_tot_T,E_tot_T,T_comp_T,E_comp_T=build_curves_vary_T(T_values,H0,W0,S0)

    figE,axesE=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
    stacked_area(axesE[0],S_values,E_comp_S,"Energy vs. S (denoising steps)","S","Energy (J)")
    stacked_area(axesE[1],scale_values*(H0*W0),E_comp_sc,"Energy vs. Resolution (H×W)","Pixels (H×W)","Energy (J)")
    stacked_area(axesE[2],T_values,E_comp_T,"Energy vs. T (frames)","T","Energy (J)")
    figE.savefig(os.path.join(args.outdir,"theory_energy_stacked.png"),dpi=args.dpi)

    figL,axesL=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
    stacked_area(axesL[0],S_values,T_comp_S,"Latency vs. S (denoising steps)","S","Latency (s)")
    stacked_area(axesL[1],scale_values*(H0*W0),T_comp_sc,"Latency vs. Resolution (H×W)","Pixels (H×W)","Latency (s)")
    stacked_area(axesL[2],T_values,T_comp_T,"Latency vs. T (frames)","T","Latency (s)")
    figL.savefig(os.path.join(args.outdir,"theory_latency_stacked.png"),dpi=args.dpi)

    wan_ds=load_dataset(args.wan_ds)
    wan_df=concat_all_splits(wan_ds)
    dfs=[]
    mape_lines=[]

    if args.wan_vary_kind_col in wan_df.columns:
        frames_df=wan_df[wan_df[args.wan_vary_kind_col]==args.wan_kind_frames].copy()
    else:
        frames_df=wan_df.copy()
    if not frames_df.empty:
        if args.width_col not in frames_df.columns: frames_df[args.width_col]=W0
        if args.height_col not in frames_df.columns: frames_df[args.height_col]=H0
        if args.steps_col not in frames_df.columns: frames_df[args.steps_col]=S0
        frames_df=frames_df.sort_values(by=args.frames_col)
        x7,e_comp7,t_comp7,e_obs7,t_obs7,e_tot7,t_tot7=build_series(frames_df,args.frames_col,args.width_col,args.height_col,args.frames_col,args.steps_col,args.obs_energy_scale,args.energy_col,args.time_col)
        plt.figure(figsize=(8,5)); ax=plt.gca()
        stacked_theory_with_practice(ax,x7,e_comp7,e_obs7,"GPU Energy vs Number of Frames\n(Theory components + Practice)","Frames","Energy (Wh)")
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"energy_vs_frames.png"),dpi=args.dpi)
        plt.figure(figsize=(8,5)); ax=plt.gca()
        stacked_theory_with_practice(ax,x7,t_comp7,t_obs7,"Generation Time vs Number of Frames\n(Theory components + Practice)","Frames","Duration (s)")
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"time_vs_frames.png"),dpi=args.dpi)
        dfs.append(frames_df)
        mape_lines.append(f"Frames:  Energy {mean_percentage_error(e_obs7,e_tot7):.2f}%, Latency {mean_percentage_error(t_obs7,t_tot7):.2f}%")

    if args.wan_vary_kind_col in wan_df.columns:
        steps_df=wan_df[wan_df[args.wan_vary_kind_col]==args.wan_kind_steps].copy()
    else:
        steps_df=wan_df.copy()
    if not steps_df.empty:
        if args.width_col not in steps_df.columns: steps_df[args.width_col]=W0
        if args.height_col not in steps_df.columns: steps_df[args.height_col]=H0
        if args.frames_col not in steps_df.columns: steps_df[args.frames_col]=T0
        steps_df=steps_df[steps_df[args.steps_col]%4==0].copy()
        steps_df=steps_df.sort_values(by=args.steps_col)
        x9,e_comp9,t_comp9,e_obs9,t_obs9,e_tot9,t_tot9=build_series(steps_df,args.steps_col,args.width_col,args.height_col,args.frames_col,args.steps_col,args.obs_energy_scale,args.energy_col,args.time_col)
        plt.figure(figsize=(8,5)); ax=plt.gca()
        stacked_theory_with_practice(ax,x9,e_comp9,e_obs9,"GPU Energy vs Denoising Steps\n(Theory components + Practice)","Denoising Steps","Energy (Wh)")
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"energy_vs_steps.png"),dpi=args.dpi)
        plt.figure(figsize=(8,5)); ax=plt.gca()
        stacked_theory_with_practice(ax,x9,t_comp9,t_obs9,"Generation Time vs Denoising Steps\n(Theory components + Practice)","Denoising Steps","Duration (s)")
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"time_vs_steps.png"),dpi=args.dpi)
        dfs.append(steps_df)
        mape_lines.append(f"Steps:   Energy {mean_percentage_error(e_obs9,e_tot9):.2f}%, Latency {mean_percentage_error(t_obs9,t_tot9):.2f}%")

    if args.wan_vary_kind_col in wan_df.columns:
        res_df=wan_df[wan_df[args.wan_vary_kind_col]==args.wan_kind_res].copy()
    else:
        res_df=wan_df.copy()
    if not res_df.empty:
        if args.frames_col not in res_df.columns: res_df[args.frames_col]=T0
        if args.steps_col not in res_df.columns: res_df[args.steps_col]=S0
        res_df["resolution"]=res_df[args.width_col]*res_df[args.height_col]
        res_df=res_df.sort_values(by="resolution")
        x8,e_comp8,t_comp8,e_obs8,t_obs8,e_tot8,t_tot8=build_series(res_df,"resolution",args.width_col,args.height_col,args.frames_col,args.steps_col,args.obs_energy_scale,args.energy_col,args.time_col)
        plt.figure(figsize=(8,5)); ax=plt.gca()
        stacked_theory_with_practice(ax,x8,e_comp8,e_obs8,"GPU Energy vs Resolution\n(Theory components + Practice)","Resolution (pixels)","Energy (Wh)")
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"energy_vs_resolution.png"),dpi=args.dpi)
        plt.figure(figsize=(8,5)); ax=plt.gca()
        stacked_theory_with_practice(ax,x8,t_comp8,t_obs8,"Generation Time vs Resolution\n(Theory components + Practice)","Resolution (pixels)","Duration (s)")
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"time_vs_resolution.png"),dpi=args.dpi)
        dfs.append(res_df)
        mape_lines.append(f"Res:     Energy {mean_percentage_error(e_obs8,e_tot8):.2f}%, Latency {mean_percentage_error(t_obs8,t_tot8):.2f}%")

    if mape_lines:
        with open(os.path.join(args.outdir,"mape.txt"),"w") as f:
            for line in mape_lines:
                f.write(line+"\n")
        for line in mape_lines:
            print(line)

    if len(dfs)>0:
        build_args=dict(N=N,d=d,f_mlp=f_mlp,m_text=m_text)
        mu_hat,b_hat,R2=estimate_mu_and_overhead_from_dfs(dfs,theta_peak,build_args,args.time_col)
        with open(os.path.join(args.outdir,"fit_mu_overhead.txt"),"w") as f:
            f.write(f"mu={mu_hat:.3f}\n")
            f.write(f"latency_overhead={b_hat:.3f}s\n")
            f.write(f"R2={R2:.4f}\n")
        print(f"mu={mu_hat:.3f}, latency_overhead={b_hat:.3f}s, R2={R2:.4f}")

    models_ds=load_dataset(args.models_ds)
    models_df=concat_all_splits(models_ds)
    if not models_df.empty:
        plot_models_bars(models_df,args.outdir,args.dpi,args.energy_col,args.time_col,args.models_model_col)

if __name__=="__main__":
    main()
