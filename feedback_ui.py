import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Toplevel, Frame, Label, BOTH, ttk

def show_feedback_window(parent, trunk_df, shank_df,
                         trunk_signals, shank_signals,
                         time_vector, results_dict, segment_id=None):
    """
    終端フィードバック画面を表示（静的セグメント表示）。
    """
    win = Toplevel(parent)
    win.title("解析結果フィードバック")
    win.geometry("900x800")

    # データコピー
    df_trunk = trunk_df.copy()
    df_shank = shank_df.copy()

    # 指定セグメントでフィルタ (segment_id が渡されていれば)
    if segment_id is not None:
        df_trunk = df_trunk[df_trunk['Trial_ID'] == segment_id]
        df_shank = df_shank[df_shank['Trial_ID'] == segment_id]

    tv = time_vector.copy()
    ts = {k: np.array(v) for k, v in trunk_signals.items()}
    ss = {k: np.array(v) for k, v in shank_signals.items()}

    # 時間範囲を体幹ICで揃える
    if not df_trunk.empty:
        t0, t1 = df_trunk['IC_Time'].min(), df_trunk['IC_Time'].max()
        mask = (tv >= t0) & (tv <= t1)
        tv = tv[mask]
        for key in ('AP', 'VT', 'ML'):
            ts[key] = ts[key][mask]
        for leg in ('L', 'R'):
            ss[leg] = ss[leg][mask]
        df_shank = df_shank[(df_shank['IC_Time'] >= t0) & (df_shank['IC_Time'] <= t1)]

    # プロット領域クリア＆描画
    for w in win.winfo_children():
        if isinstance(w, FigureCanvasTkAgg):
            w.get_tk_widget().destroy()
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax in axes:
        ax.title.set_fontsize(14)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.tick_params(labelsize=10)

    # 体幹 AP 加速度 + IC
    axes[0].plot(tv, ts['AP'], label='Trunk AP Acc', alpha=0.7)
    if not df_trunk.empty:
        idx = np.clip(np.searchsorted(tv, df_trunk['IC_Time'].values, 'left'),
                      0, len(tv)-1)
        axes[0].plot(tv[idx], ts['AP'][idx], 'ro', ms=5, label='IC')
    axes[0].set_title("体幹 AP 加速度")
    axes[0].legend(loc='upper right'); axes[0].grid(True)

    # 下腿 Gyro Z + IC/FO
    axes[1].plot(tv, ss['L'], label='L Gyro Z', alpha=0.7)
    axes[1].plot(tv, ss['R'], label='R Gyro Z', linestyle='--', alpha=0.7)
    if not df_shank.empty:
        for leg, m in [('L', 'ro'), ('R', 'bo')]:
            ev = df_shank[df_shank['Leg'] == leg]
            idx_ic = np.clip(np.searchsorted(tv, ev['IC_Time'].values, 'left'),
                             0, len(tv)-1)
            idx_fo = np.clip(np.searchsorted(tv, ev['FO_Time'].dropna().values, 'left'),
                             0, len(tv)-1)
            axes[1].plot(tv[idx_ic], ss[leg][idx_ic], m, ms=5, linestyle='None', label=f'IC({leg})')
            axes[1].plot(tv[idx_fo], ss[leg][idx_fo], 'x', ms=7, markeredgewidth=2, label=f'FO({leg})')
    axes[1].set_title("下腿 Gyro Z")
    axes[1].legend(loc='upper right'); axes[1].grid(True)

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(side='top', fill=BOTH, expand=False)
    plt.close(fig)

    # パラメータテーブル表示
    for w in win.winfo_children():
        if isinstance(w, ttk.Treeview):
            w.destroy()
    frame = Frame(win)
    frame.pack(side='bottom', fill=BOTH, expand=True, pady=0)
    tree = ttk.Treeview(frame, columns=('Parameter','Value','Parameter2','Value2'),
                        show='headings', height=15)
    style = ttk.Style(win)
    style.configure("Treeview", font=('TkDefaultFont',12))
    style.configure("Treeview.Heading", font=('TkDefaultFont',12,'bold'))
    for col in ('Parameter','Value','Parameter2','Value2'):
        tree.heading(col, text=col)
        tree.column(col, width=(100 if 'Parameter' in col else 80),
                    anchor=('e' if 'Parameter' in col else 'w'))

    pci_keys = ['PCI','P_phi_ABS','phi_ABS_deg','phi_CV_percent','std_phase_deg']
    hr_keys = ['HR_AP','HR_VT','HR_ML','iHR_AP','iHR_VT','iHR_ML']
    sym_keys = sorted([k for k in results_dict if k.startswith('SI_')])
    mean_keys = [k for k in ['Mean_Stance_Time_L_s','Mean_Stance_Time_R_s',
                             'Mean_Stride_Time_L_s','Mean_Stride_Time_R_s',
                             'Mean_Swing_Time_L_s','Mean_Swing_Time_R_s',
                             'Cadence_steps_per_min'] if k in results_dict]
    groups = [("PCI関連", pci_keys), ("HR関連", hr_keys),
              ("左右差(SI)", sym_keys), ("平均値・Cadence", mean_keys)]
    for label, keys in groups:
        if not keys: continue
        tree.insert('', 'end', values=('', '', '', ''))
        tree.insert('', 'end', values=(label, '', '', ''))
        for i in range(0, len(keys), 2):
            k1, v1 = keys[i], results_dict[keys[i]]
            t1 = f"{v1:.3f}" if isinstance(v1, (float, np.floating)) else str(v1)
            if i+1 < len(keys):
                k2, v2 = keys[i+1], results_dict[keys[i+1]]
                t2 = f"{v2:.3f}" if isinstance(v2,(float,np.floating)) else str(v2)
            else:
                k2, t2 = '', ''
            tree.insert('', 'end', values=(k1, t1, k2, t2))
    tree.pack(fill=BOTH, expand=True)