#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from plotly.subplots import make_subplots
from pypdf import PdfReader

from longcat.io import ROOT, load_config, sha256_file

IVORY="#f6f1e7"; TEAL="#0f766e"; MINT="#8bd4c8"; CHARCOAL="#263238"; SAND="#d8a15d"; GRID="#c9c0ae"


def finish(fig: go.Figure, path: Path, width: int, height: int, legend: bool = False) -> None:
    fig.update_layout(width=width,height=height,paper_bgcolor=IVORY,plot_bgcolor=IVORY,font=dict(family="Avenir Next, Arial",color=CHARCOAL,size=12),margin=dict(l=45,r=20,t=35,b=40),showlegend=legend)
    fig.update_xaxes(gridcolor=GRID,zerolinecolor=GRID);fig.update_yaxes(gridcolor=GRID,zerolinecolor=GRID)
    fig.write_image(path,format="svg",width=width,height=height,scale=1)


def cover_system(path: Path) -> None:
    fig=go.Figure(go.Sankey(arrangement="fixed",node=dict(pad=18,thickness=18,line=dict(color=IVORY,width=1),color=[TEAL,TEAL,CHARCOAL,SAND,SAND],label=["EAGLE-I telemetry","Issue-time boundary","Renewal state","Task A · 48 h","Task B · 6 h"],x=[.02,.27,.52,.88,.88],y=[.5,.5,.5,.27,.73]),link=dict(source=[0,1,2,2],target=[1,2,3,4],value=[2,2,1,1],color=["rgba(15,118,110,.35)","rgba(15,118,110,.35)","rgba(216,161,93,.42)","rgba(216,161,93,.42)"])))
    fig.add_annotation(x=.52,y=.04,text="only observations timestamped ≤ issue time",showarrow=False,font=dict(size=11,color=TEAL));finish(fig,path,1000,430)


def coverage(path: Path, exploration: pd.DataFrame) -> None:
    fig=go.Figure();colors=[GRID,MINT,TEAL]
    for year,color in zip((2023,2024,2025),colors):
        block=exploration[exploration.year==year];fig.add_bar(x=block.county,y=block.coverage_pct,name=str(year) + (" through Aug" if year==2025 else ""),marker_color=color)
    fig.update_layout(barmode="group",legend=dict(orientation="h",y=1.15,x=0));fig.update_yaxes(title="observed quarter-hours (%)",range=[70,100]);finish(fig,path,920,360,True)


def exploration_plot(path: Path, exploration: pd.DataFrame) -> None:
    x=exploration.sort_values('fips_code').drop_duplicates('fips_code');ratio=np.minimum(120,x.p99_customers_out/np.maximum(x.median_customers_out,1))
    fig=make_subplots(rows=1,cols=2,subplot_titles=("15-minute log persistence ↑","p99 / median magnitude"))
    fig.add_bar(x=x.county,y=x.lag15_log_correlation_all_years,marker_color=TEAL,row=1,col=1,texttemplate="%{y:.2f}",textposition="outside")
    fig.add_bar(x=x.county,y=ratio,marker_color=SAND,row=1,col=2,texttemplate="%{y:.0f}×",textposition="outside")
    fig.update_yaxes(range=[0,1.02],row=1,col=1);fig.update_yaxes(range=[0,132],row=1,col=2);finish(fig,path,820,500)


def nodes(path: Path, model: bool=False) -> None:
    if model:
        labels=["Issue-time state","Scale + encode","Log-count Ridge A","Log-count Ridge B","20% renewal","MAE calibrator","MCC projection","predicted_x"];x=[0,1,2,2,3,4,5,6];y=[0,0,.45,-.45,-.45,0,0,0];colors=[TEAL,CHARCOAL,TEAL,TEAL,SAND,SAND,CHARCOAL,TEAL];edges=[(0,1),(1,2),(1,3),(3,4),(2,5),(4,5),(5,6),(6,7)]
    else:
        labels=["Now","1/6/24 h state","7 d tail state","1/7/14 d recurrence","lead + cycles","regional median","task head"];x=[0,1,1,2,2,3,4];y=[0,.42,-.42,.42,-.42,0,0];colors=[TEAL,TEAL,TEAL,SAND,CHARCOAL,SAND,TEAL];edges=[(0,1),(0,2),(1,3),(2,4),(3,5),(4,5),(5,6)]
    fig=go.Figure(go.Scatter(x=x,y=y,mode="markers+text",text=labels,textposition="bottom center",marker=dict(size=38,color=colors,line=dict(color=IVORY,width=2))))
    for a,b in edges:fig.add_annotation(x=x[b],y=y[b],ax=x[a],ay=y[a],xref='x',yref='y',axref='x',ayref='y',showarrow=True,arrowhead=2,arrowwidth=2,arrowcolor=GRID)
    fig.update_xaxes(visible=False,range=[-.25,6.25] if model else [-.25,4.25]);fig.update_yaxes(visible=False,range=[-.85,.85]);finish(fig,path,1000,310)


def validation_plot(path: Path, metrics: pd.DataFrame) -> None:
    block=metrics[(metrics.fold=='2024_forward') & metrics.method.isin(['weekly_routine','last_observation','tideglass_full'])].copy();labels={'weekly_routine':'Weekly routine','last_observation':'Last observation','tideglass_full':'Tideglass'};colors={'weekly_routine':GRID,'last_observation':CHARCOAL,'tideglass_full':TEAL}
    fig=make_subplots(rows=1,cols=2,subplot_titles=("Task A · MAE × 10⁵ ↓","Task B · MAE × 10⁵ ↓"))
    for col,task in enumerate('AB',start=1):
        t=block[block.task_id==task];fig.add_bar(x=[labels[x] for x in t.method],y=t.mae*1e5,marker_color=[colors[x] for x in t.method],text=[f"{x:.2f}" for x in t.mae*1e5],textposition="outside",row=1,col=col)
    fig.update_yaxes(rangemode='tozero');finish(fig,path,900,430)


def causal_audit(path: Path) -> None:
    fig=go.Figure();fig.add_trace(go.Scatter(x=[-168,-24,0],y=[2,2,2],mode='lines+markers',line=dict(color=TEAL,width=10),marker=dict(size=10)));fig.add_trace(go.Scatter(x=[1,48],y=[1,1],mode='lines+markers',line=dict(color=SAND,width=10),marker=dict(size=10)));fig.add_trace(go.Scatter(x=[.25,6],y=[0,0],mode='lines+markers',line=dict(color=CHARCOAL,width=10),marker=dict(size=10)))
    fig.add_vline(x=0,line_color=CHARCOAL,line_width=2);fig.add_annotation(x=0,y=2.5,text='issue time',showarrow=False);fig.update_yaxes(tickvals=[0,1,2],ticktext=['Task B targets','Task A targets','observable state'],range=[-.5,2.6]);fig.update_xaxes(title='hours relative to issue time',tickvals=[-168,-24,0,6,24,48]);finish(fig,path,690,500)


def prediction_distribution(path: Path, predictions: pd.DataFrame) -> None:
    fig=go.Figure()
    for task,color in [('A',TEAL),('B',SAND)]:
        values=np.log10(np.maximum(predictions.loc[predictions.task_id==task,'predicted_x'].to_numpy(),1e-7));fig.add_histogram(x=values,name=f'Task {task}',opacity=.65,nbinsx=35,marker_color=color,histnorm='probability density')
    fig.update_layout(barmode='overlay',legend=dict(orientation='h',y=1.18,x=0));fig.update_xaxes(title='log10 predicted ratio');fig.update_yaxes(title='density');finish(fig,path,680,300,True)


def main() -> None:
    report=ROOT/'report';figures=report/'figures';figures.mkdir(parents=True,exist_ok=True);cfg=load_config();exploration=pd.read_csv(ROOT/'artifacts'/'exploration_by_county.csv',dtype={'fips_code':str});metrics=pd.read_csv(ROOT/'artifacts'/'validation_metrics.csv');predictions=pd.read_csv(ROOT/'predictions.csv',dtype={'fips_code':str})
    cover_system(figures/'cover_system.svg');coverage(figures/'coverage.svg',exploration);exploration_plot(figures/'exploration.svg',exploration);nodes(figures/'feature_layers.svg');nodes(figures/'model_pipeline.svg',True);validation_plot(figures/'validation.svg',metrics);causal_audit(figures/'causal_audit.svg');prediction_distribution(figures/'prediction_distribution.svg',predictions)
    roles={"17031":"scale anchor + Midwest contrast","34003":"dense Northeast contrast","36047":"large urban recurrence","36081":"large urban recurrence","36103":"coastal tail behavior"};counties=[{**c,'customers_fmt':f"{c['customers']:,}",'role':roles[c['fips_code']]} for c in cfg['counties']]
    v=metrics[(metrics.fold=='2024_forward') & metrics.method.isin(['last_observation','tideglass_full'])]
    def gain(task,metric):
        x=v[v.task_id==task].set_index('method')[metric];return f"{100*(x['last_observation']-x['tideglass_full'])/x['last_observation']:.1f}%"
    display=[];names={'weekly_routine':'Weekly routine','last_observation':'Last observation','uncalibrated_renewal':'Before calibration','tideglass_full':'Tideglass full'}
    selected=metrics[metrics.fold.eq('2024_forward') & metrics.method.isin(names)]
    for _,r in selected.iterrows():display.append({'method':names[r.method],'task_id':r.task_id,'rows':f"{int(r.rows):,}",'mae':f"{r.mae:.6f}",'rmse':f"{r.rmse:.6f}",'tail':f"{r.top_decile_mae:.6f}",'corr':f"{r.pearson:.3f}"})
    corr=exploration.drop_duplicates('fips_code').lag15_log_correlation_all_years.median();env=Environment(loader=FileSystemLoader(report),undefined=StrictUndefined,autoescape=True)
    html=env.get_template('report_template.html.j2').render(team_name=cfg['project']['team_name'],participant_name=cfg['project']['participant_name'],contact=cfg['project']['contact'],counties=counties,median_corr=f"{corr:.3f}",validation_rows=display,a_mae_gain=gain('A','mae'),a_rmse_gain=gain('A','rmse'),b_mae_gain=gain('B','mae'),b_rmse_gain=gain('B','rmse'),prediction_hash=sha256_file(ROOT/'predictions.csv'))
    html_path=report/'report.html';html_path.write_text(html,encoding='utf-8');target=ROOT/'LongCat_Challenge2.pdf';chrome=Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    subprocess.run([str(chrome),'--headless','--disable-gpu','--no-pdf-header-footer',f'--print-to-pdf={target}',html_path.resolve().as_uri()],check=True,capture_output=True,text=True)
    pages=len(PdfReader(target).pages)
    if not 3<=pages<=8:raise RuntimeError(f'report page count {pages}')
    print(f'Built {target} ({pages} pages)')


if __name__=='__main__':main()
