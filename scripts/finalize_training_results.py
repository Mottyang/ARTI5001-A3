"""Freeze final experimental evidence and plot existing logs; never train."""
from pathlib import Path
import csv,json,hashlib,statistics,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluations/final_1k_vs_2k'
OUT.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result_path=ROOT/'comparisons/training_1k_vs_2k/results.json'
results=json.loads(result_path.read_text(encoding='utf-8'))
preserved=[]
for name in ('expA','expB'):
    for folder in (ROOT/'checkpoints'/name,ROOT/'checkpoints'/name/'stage_2k'):
        for file in ('best.pt','last.pt'):
            p=folder/file;preserved.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
continuation=json.loads((ROOT/'data_manifests/formal/continuation_2k.json').read_text(encoding='utf-8'))
assert all(sha(ROOT/x['path'])==x['sha256'] for x in continuation['preserved_1k_files'])

observation={
    'id':'human_1k_vs_2k_no_clear_gain_20261028','date_recorded':'2026-10-28',
    'source':'user manual listening in this conversation',
    'comparison':'comparisons/training_1k_vs_2k/listen.html',
    'user_statement':'人工试听后，发现并没有明显进步，有些地方反而有退化。',
    'judgment':'No obvious overall improvement from 1k to 2k; some passages regressed.',
    'scope':'Overall feedback on the 1k/2k comparison; specific sample/model locations were not provided.',
    'numeric_score':None,'sample_ids':None,'per_model_scores':None,'independent_rater_count':1,
    'limitations':'Do not interpret as three independent raters or as every model/sample regressing.'}
human_path=ROOT/'evaluations/human_listening_observations.json'
human=json.loads(human_path.read_text(encoding='utf-8'))
human['observations']=[x for x in human['observations'] if x['id']!=observation['id']]+[observation]
human_path.write_text(json.dumps(human,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'human_feedback.json').write_text(json.dumps(observation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (OUT/'human_feedback.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(observation));writer.writeheader();writer.writerow(observation)

metrics={};log_hashes={};combined=[]
for name in ('expA','expB'):
    rows=[]
    for stage,path in [('1k',ROOT/'logs'/name/'metrics.csv'),('2k_continuation',ROOT/'logs'/name/'stage_2k/metrics.csv')]:
        log_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
        for row in csv.DictReader(path.open(encoding='utf-8',newline='')):
            parsed={'experiment':name,'stage':stage,'step':int(row['step']),
                    **{k:float(row[k]) for k in ['train_loss','mel_loss','gate_loss','gradient_norm']},
                    'validation_loss':float(row['validation_loss']) if row['validation_loss'] else None}
            rows.append(parsed);combined.append(parsed)
    assert [x['step'] for x in rows]==list(range(1,2001))
    assert all(np.isfinite(x['train_loss']) for x in rows)
    metrics[name]=rows
with (OUT/'training_metrics_0_to_2k.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(combined[0]));writer.writeheader();writer.writerows(combined)

table=[]
for name,lr in [('expA',1e-5),('expB',5e-5)]:
    for steps in (1000,2000):
        variant=f'{name}_{steps//1000}k'
        val=next(x['loss'] for x in results['audits'][name]['validation'] if x['step']==steps)
        ref=results['summary']['reference'][variant];custom=results['summary']['custom'][variant]
        table.append({'experiment':name,'learning_rate':lr,'endpoint_steps':steps,
            'validation_loss':val,'reference_word_errors':ref['word_errors'],'reference_word_count':ref['reference_words'],
            'reference_wer':ref['micro_wer'],'custom_word_errors':custom['word_errors'],'custom_word_count':custom['reference_words'],
            'custom_wer':custom['micro_wer'],'reference_latency_seconds':ref['mean_latency_seconds'],
            'reference_rtf':ref['aggregate_rtf'],'custom_latency_seconds':custom['mean_latency_seconds'],
            'custom_rtf':custom['aggregate_rtf']})
with (OUT/'final_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
validation=[]
initial=json.loads((ROOT/'logs/expA/training_report.json').read_text(encoding='utf-8'))['initial_validation_loss']
validation.append({'step':0,'expA':initial,'expB':initial})
for a,b in zip(results['audits']['expA']['validation'],results['audits']['expB']['validation']):
    assert a['step']==b['step'];validation.append({'step':a['step'],'expA':a['loss'],'expB':b['loss']})
with (OUT/'validation_curve_data.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['step','expA','expB']);w.writeheader();w.writerows(validation)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.bbox':'tight','svg.fonttype':'none','pdf.fonttype':42})
colors={'expA':'#166d9c','expB':'#b45521'}
def validation_panel(ax,zoom=False):
    for name in ('expA','expB'):
        ax.plot([x['step'] for x in validation],[x[name] for x in validation],marker='o',ms=4,lw=2,color=colors[name],label=name)
        best=results['audits'][name]
        ax.scatter([best['best_step_within_2k']],[best['best_loss_within_2k']],marker='*',s=180,color=colors[name],edgecolor='white',zorder=5)
    ax.axvline(1000,color='#68717c',ls='--',lw=1,label='1k boundary')
    ax.set(xlabel='Total optimizer updates',ylabel='Teacher-forced validation loss',xlim=(-20,2050),
           title='Validation loss: exact checkpoints; stars = best within 2k')
    if zoom:ax.set_ylim(.370,.390)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=9));ax.grid(alpha=.22);ax.legend(frameon=False)

fig,axes=plt.subplots(1,2,figsize=(13,4.6))
for name,rows in metrics.items():
    steps=np.array([x['step'] for x in rows]);loss=np.array([x['train_loss'] for x in rows])
    axes[0].plot(steps,loss,color=colors[name],alpha=.13,lw=.6)
    smoothed=np.convolve(loss,np.ones(50)/50,mode='valid')
    axes[0].plot(steps[49:],smoothed,color=colors[name],lw=1.8,label=f'{name}: trailing 50-step mean')
axes[0].axvline(1000,color='#68717c',ls='--',lw=1)
axes[0].set(title='Training loss: raw batches + trailing mean',xlabel='Total optimizer updates',ylabel='Total training loss',xlim=(0,2000))
axes[0].grid(alpha=.22);axes[0].legend(frameon=False)
validation_panel(axes[1]);fig.tight_layout()
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'training_curves.{ext}',dpi=220)
plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.8));validation_panel(ax,zoom=True);fig.tight_layout()
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'validation_curve_zoom.{ext}',dpi=220)
plt.close(fig)

lines=['# 1k 与 2k 最终结果及实验结论','',
'整理日期：2026-10-28。用户已决定结束训练，不继续增加步数。全部已有检查点、日志、音频和历史结果保留。','',
'## 人工反馈','',
'> 人工试听后，发现并没有明显进步，有些地方反而有退化。','',
'这是用户对整体1k/2k试听的定性反馈，没有提供具体句子、位置或逐模型分数；不能据此声称每个模型、每句话都退化，也不能伪造三名组员的独立评分。','',
'## 固定条件','',
'相同2000条训练样本、1310条验证样本、batch=4、seed=42、Adam、梯度裁剪与预处理。A学习率1e-5，B为5e-5。两组各从自己的1k模型/Adam/随机状态继续到2k，属于同一训练轨迹延长，不是四次独立实验。评估使用相同10条验证参照+10条自造句；数字/金额修复启用，缩写展开关闭。','',
'## 精确终点比较','',
'| 模型 | 学习率 | 步数 | 验证损失 | 真实参照WER | 自造句WER | 真实参照RTF | 自造句RTF |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for row in table:
    lines.append(f'| {row["experiment"]} | {row["learning_rate"]:.0e} | {row["endpoint_steps"]} | {row["validation_loss"]:.6f} | {row["reference_wer"]:.2%} ({row["reference_word_errors"]}/148) | {row["custom_wer"]:.2%} ({row["custom_word_errors"]}/120) | {row["reference_rtf"]:.5f} | {row["custom_rtf"]:.5f} |')
lines+=['','原始模型的辅助WER参考：真实参照10条4.73%（7/148），自造10句5.83%（7/120）。两组评估集分别汇总，不合并平均。','',
'## 最佳检查点与预算终点','',
'| 实验 | 2k预算内最低验证损失步数 | 最低验证损失 |','|---|---:|---:|']
for name,a in results['audits'].items():lines.append(f'| {name} | {a["best_step_within_2k"]} | {a["best_loss_within_2k"]:.6f} |')
lines+=['','A的最低验证损失在1750步，不是2000步；B的最低值仍在1000步。1k/2k试听页比较精确终点，不把1750步冒充2k。1750步A已核验重载，但本次人工反馈来自1k/2k页面，不能推断它已获得独立试听验证。','',
'## 曲线解读','',
'- 训练曲线同时保留原始批次损失与50步后向滑动均值。不同批次难度不同，原始曲线波动不等于训练失败。',
'- 验证曲线显示A在1k后改善有限且不单调；B的所有新增保存点均未优于原1k。',
'- 两组日志各连续覆盖1到2000步；新增1000步损失及梯度均为有限值，检查点严格重载成功。',
'- 验证使用teacher forcing，prenet仍有随机性；微小损失差异不等于稳定的听感收益。','',
'## 最终结论（限定于本次设置）','',
'1. 较小学习率A在当前验证损失上优于B，但没有充分证据证明其听感全面更好。',
'2. A从1k到2k终点验证损失只略降（0.373902→0.373427），真实参照WER不变，自造句仅净少一个ASR词错误；与用户反馈结合，不支持“明显音质进步”的结论。',
'3. B从1k到2k验证损失升高（0.383728→0.387289），两类样本各净多一个ASR词错误；追加预算未带来一致收益。',
'4. 在本项目的数据量、模型、学习率和训练预算下，增加到2k没有产生一致、可感知的语音质量改善。不能推广为所有Tacotron2训练超过1k都无效。',
'5. 按用户决定停止继续训练，转入评估、错误分析和提交整理。最终使用版本不能仅靠最低损失确定；A1k可作为成本较低的微调候选，原始模型保留为基线，A1750仅标为验证损失最优候选。','',
'## 指标与证据限制','',
'ASR不是自然度评分；单词拼写、词边界和识别器错误会影响WER。RTF由生成时间和输出时长共同决定，三次重复计时仍受系统负载影响，不能从小幅变化认定计算效率提高。验证集已用于开发诊断，不是未见最终测试；LJSpeech预训练重叠、单训练种子和少量句子限制仍需说明。未虚构MOS、统计显著性或组员评分。','',
'## 文件','',
'- final_comparison.csv：精确1k/2k终点对比，可导入Excel。',
'- training_curves.png/.svg/.pdf：训练和验证双图，PNG用于预览，SVG/PDF用于导出。',
'- validation_curve_zoom.png/.svg/.pdf：验证损失放大图，纵轴明确截断至0.370–0.390；步0数值在此放大图外，完整双图包含步0。',
'- training_metrics_0_to_2k.csv：4000行实际训练批次数据（两组各2000行；没有伪造step0训练损失）。',
'- validation_curve_data.csv：真实验证保存点，包括共同的训练前验证值。',
'- human_feedback.json/.csv：用户原话与真实评价范围。',
'- evidence_manifest.json：引用文件和检查点哈希。',
'- ../../comparisons/training_1k_vs_2k/listen.html：已有试听页面，保留原音频。']
(OUT/'FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
snapshot={'date':'2026-10-28','training_decision':'Stop additional training at user request after 2k comparison',
    'feedback':observation,'endpoint_comparison':table,
    'best_within_2k':{n:{'step':a['best_step_within_2k'],'loss':a['best_loss_within_2k']} for n,a in results['audits'].items()},
    'checkpoint_preservation':preserved,'source_logs_sha256':log_hashes,'source_comparison_sha256':sha(result_path),
    'figure_notes':'Raw batch losses plus trailing 50-step mean. Full validation panel includes step0; zoom panel explicitly clips y to .370-.390.',
    'no_new_training_or_audio_generation':True,'human_scores_invented':False}
(OUT/'final_results.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for item in preserved:assert sha(ROOT/item['path'])==item['sha256']
(OUT/'evidence_manifest.json').write_text(json.dumps({'checkpoints':preserved,'logs':log_hashes,'source_comparison_sha256':sha(result_path)},indent=2)+'\n',encoding='utf-8')
page=ROOT/'comparisons/training_1k_vs_2k/listen.html'
s=page.read_text(encoding='utf-8')
notice='<p style="padding:14px;background:#fff0d5"><strong>人工试听与收尾决定：</strong>用户反馈“没有明显进步，有些地方反而退化”，已决定停止追加训练。未提供逐句或逐模型数值评分。<a href="../../evaluations/final_1k_vs_2k/FINAL_RESULTS.md">查看最终结果和曲线记录</a></p>'
heading='<h1>A/B：同一句的 1k 与 2k 训练结果</h1>'
if '人工试听与收尾决定' not in s:page.write_text(s.replace(heading,heading+notice),encoding='utf-8')
record=ROOT/'PROJECT_RECORD.md';s=record.read_text(encoding='utf-8')
note='> 最终收尾（2026-10-28）：用户人工试听未发现明显进步，部分退化，已决定不继续训练。最终表格、曲线和结论见 evaluations/final_1k_vs_2k/FINAL_RESULTS.md。原检查点和结果保留。\n\n'
if not s.startswith('> 最终收尾'):record.write_text(note+s+'\n\n# 最终人工反馈与停止训练\n\n人工试听：没有明显进步，有些地方反而退化。用户决定停止追加训练，保留1k/2k全部检查点及结果，转入提交准备。详见 evaluations/final_1k_vs_2k/FINAL_RESULTS.md。没有补造逐模型分数。\n',encoding='utf-8')
readme=ROOT/'README.md';s=readme.read_text(encoding='utf-8');note='最终实验收尾：[evaluations/final_1k_vs_2k/FINAL_RESULTS.md](evaluations/final_1k_vs_2k/FINAL_RESULTS.md)。训练已停止，检查点保留，包含对比表、曲线和人工反馈。\n\n'
if not s.startswith('最终实验收尾'):readme.write_text(note+s,encoding='utf-8')
print(json.dumps({'output':str(OUT),'preserved_checkpoint_files':len(preserved),'training_rows':len(combined),'human_observation_id':observation['id'],'no_training':True},indent=2))
