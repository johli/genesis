import isolearn.keras as iso

import numpy as np

import pandas as pd

import scipy.sparse as sp
import scipy.io as spio

import matplotlib.pyplot as plt

import seaborn as sns

def plot_logo_w_hexamer_scores(pwm, iso_pred, cut_pred, hexamer_scores, cse_start_pos=70, annotate_peaks=None, sequence_template=None, figsize=(12, 3), width_ratios=[1, 7], logo_height=1.0, usage_unit='log', plot_start=0, plot_end=164, save_figs=False, fig_name=None, fig_dpi=300) :

    n_samples = pwm.shape[0]

    #Slice according to seq trim index
    pwm = pwm[:, plot_start: plot_end, :]
    cut_pred = cut_pred[:, plot_start: plot_end]
    sequence_template = sequence_template[plot_start: plot_end]

    iso_pred = np.mean(iso_pred, axis=0)
    cut_pred = np.mean(cut_pred, axis=0)
    pwm = np.sum(pwm, axis=0)

    pwm += 0.0001
    for j in range(0, pwm.shape[0]) :
        pwm[j, :] /= np.sum(pwm[j, :])

    entropy = np.zeros(pwm.shape)
    entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(pwm[pwm > 0])
    entropy = np.sum(entropy, axis=1)
    conservation = 2 - entropy

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(2, 2, width_ratios=[width_ratios[0], width_ratios[-1]], height_ratios=[1, 1])

    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])

    plt.sca(ax0)
    plt.axis('off')
    plt.sca(ax2)
    plt.axis('off')


    annot_text = 'Samples = ' + str(int(n_samples))
    if usage_unit == 'log' :
        annot_text += '\nUsage = ' + str(round(np.log(iso_pred[0] / (1. - iso_pred[0])), 4))
    else :
        annot_text += '\nUsage = ' + str(round(iso_pred[0], 4))

    ax2.text(0.99, 0.5, annot_text, horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes, color='black', fontsize=12, weight="bold")

    l2, = ax1.plot(np.arange(plot_end - plot_start), cut_pred, linewidth=3, linestyle='-', label='Predicted', color='red', alpha=0.7)

    if annotate_peaks is not None :
        objective_pos = 0
        if annotate_peaks == 'max' :
            objective_pos = np.argmax(cut_pred)
        else :
            objective_pos = annotate_peaks - plot_start

        text_x, text_y, ha = -30, -5, 'right'
        if objective_pos < 30 :
            text_x, text_y, ha = 30, -5, 'left'

        annot_text = '(CSE+' + str(objective_pos + plot_start - (cse_start_pos + 6) + 0) + ') ' + str(int(round(cut_pred[objective_pos] * 100, 0))) + '% Cleavage'
        ax1.annotate(annot_text, xy=(objective_pos, cut_pred[objective_pos]), xycoords='data', xytext=(text_x, text_y), ha=ha, fontsize=10, weight="bold", color='black', textcoords='offset points', arrowprops=dict(connectionstyle="arc3,rad=-.1", headlength=8, headwidth=8, shrink=0.15, width=1.5, color='black'))

    plt.sca(ax1)

    plt.xlim((0, plot_end - plot_start))
    #plt.ylim((0, 2))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.legend(handles=[l2], fontsize=12, prop=dict(weight='bold'), frameon=False)
    plt.axis('off')

    height_base = (1.0 - logo_height) / 2.

    for j in range(0, pwm.shape[0]) :
        sort_index = np.argsort(pwm[j, :])

        for ii in range(0, 4) :
            i = sort_index[ii]

            nt_prob = pwm[j, i] * conservation[j]

            nt = ''
            if i == 0 :
                nt = 'A'
            elif i == 1 :
                nt = 'C'
            elif i == 2 :
                nt = 'G'
            elif i == 3 :
                nt = 'T'

            color = None
            if sequence_template[j] != 'N' :
                color = 'black'

            if ii == 0 :
                letterAt(nt, j + 0.5, height_base, nt_prob * logo_height, ax3, color=color)
            else :
                prev_prob = np.sum(pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
                letterAt(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, ax3, color=color)
            
            text_x, text_y, ha = -30, -5, 'right'
            if objective_pos < 30 :
                text_x, text_y, ha = 30, -5, 'left'

            if j < len(hexamer_scores) :
                #annot_text = str(hexamer_scores[j][0]) + " = " + str(round(hexamer_scores[j][1], 2))
                
                annot_text = str(hexamer_scores[j][1])
                ax3.text(j + 0.25, 2.0, annot_text, size=50, rotation=90., ha="left", va="bottom", fontsize=8, weight="bold", color='black')

    plt.sca(ax3)

    plt.xlim((0, plot_end - plot_start))
    plt.ylim((0, 2.5))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    ax3.axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)


    for axis in fig.axes :
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

    plt.tight_layout()

    if save_figs :
        plt.savefig(fig_name + '.png', transparent=True, dpi=fig_dpi)
        plt.savefig(fig_name + '.svg')
        plt.savefig(fig_name + '.eps')

    plt.show()
    #plt.close()

mer6_dict = {}
i = 0
for b1 in ['A', 'C', 'G', 'T'] :
    for b2 in ['A', 'C', 'G', 'T'] :
        for b3 in ['A', 'C', 'G', 'T'] :
            for b4 in ['A', 'C', 'G', 'T'] :
                for b5 in ['A', 'C', 'G', 'T'] :
                    for b6 in ['A', 'C', 'G', 'T'] :
                        mer6_dict[b1 + b2 + b3 + b4 + b5 + b6] = i
                        i += 1

def get_hexamer_pred_both_regions(seq) :
    preds = []
    for cell_line, w, w_0 in zip(['hek', 'hela', 'mcf7', 'cho'], [hek_both_regions_w, hela_both_regions_w, mcf7_both_regions_w, cho_both_regions_w], [hek_both_regions_w_0, hela_both_regions_w_0, mcf7_both_regions_w_0, cho_both_regions_w_0]) :
        pred = w_0
        
        region_1 = seq[5: 40]
        for j in range(0, len(region_1) - 5) :
            pred += w[mer6_dict[region_1[j: j+6]]]
        
        region_2 = seq[48: 83]
        for j in range(0, len(region_2) - 5) :
            pred += w[4096 + mer6_dict[region_2[j: j+6]]]
        
        preds.append(1. / (1. + np.exp(-pred)))
    
    return np.array(preds)

def get_hexamer_diff_scores_both_regions(seq, cell_1, cell_2) :
    scores = []
    
    w_dict = {
        'hek' : hek_both_regions_w,
        'hela' : hela_both_regions_w,
        'mcf7' : mcf7_both_regions_w,
        'cho' : cho_both_regions_w
    }
    
    w_cell_1 = w_dict[cell_1]
    w_cell_2 = w_dict[cell_2]
    
    scores.extend([('_', ''), ('_', ''), ('_', ''), ('_', ''), ('_', '')])
    
    region_1 = seq[5: 40]
    for j in range(0, len(region_1) - 5) :
        hexamer_score_cell_1 = w_cell_1[mer6_dict[region_1[j: j+6]]]
        hexamer_score_cell_2 = w_cell_2[mer6_dict[region_1[j: j+6]]]
        
        scores.append((region_1[j: j+6], str(round(hexamer_score_cell_1 - hexamer_score_cell_2, 2))))

    scores.extend([('_', ''), ('_', ''), ('_', ''), ('_', ''), ('_', ''), ('_', ''), ('_', ''), ('_', ''), ('_', ''), ('_', ''), ('_', ''), ('_', ''), ('_', '')])
    
    region_2 = seq[48: 83]
    for j in range(0, len(region_2) - 5) :
        hexamer_score_cell_1 = w_cell_1[4096 + mer6_dict[region_2[j: j+6]]]
        hexamer_score_cell_2 = w_cell_2[4096 + mer6_dict[region_2[j: j+6]]]
        
        scores.append((region_2[j: j+6], str(round(hexamer_score_cell_1 - hexamer_score_cell_2, 2))))
    
    return scores

def get_hexamer_preds_both_regions(seqs) :
    preds = np.zeros((len(seqs), 4))
    
    for i, seq in enumerate(seqs) :
        preds[i, :] = get_hexamer_pred_both_regions(seq)
    
    return preds

def get_hexamer_pred(seq) :
    preds = []
    for cell_line, w, w_0 in zip(['hek', 'hela', 'mcf7', 'cho'], [hek_w, hela_w, mcf7_w, cho_w], [hek_w_0, hela_w_0, mcf7_w_0, cho_w_0]) :
        pred = w_0
        
        region_1 = seq[5: 40]
        for j in range(0, len(region_1) - 5) :
            pred += w[mer6_dict[region_1[j: j+6]]]
        
        preds.append(1. / (1. + np.exp(-pred)))
    
    return np.array(preds)

def get_hexamer_diff_scores(seq, cell_1, cell_2) :
    scores = []
    
    w_dict = {
        'hek' : hek_w,
        'hela' : hela_w,
        'mcf7' : mcf7_w,
        'cho' : cho_w
    }
    
    w_cell_1 = w_dict[cell_1]
    w_cell_2 = w_dict[cell_2]
    
    scores.extend([('_', ''), ('_', ''), ('_', ''), ('_', ''), ('_', '')])
    region_1 = seq[5: 40]
    for j in range(0, len(region_1) - 5) :
        hexamer_score_cell_1 = w_cell_1[mer6_dict[region_1[j: j+6]]]
        hexamer_score_cell_2 = w_cell_2[mer6_dict[region_1[j: j+6]]]
        
        scores.append((region_1[j: j+6], str(round(hexamer_score_cell_1 - hexamer_score_cell_2, 2))))
    
    return scores

def get_hexamer_preds(seqs) :
    preds = np.zeros((len(seqs), 4))
    
    for i, seq in enumerate(seqs) :
        preds[i, :] = get_hexamer_pred(seq)
    
    return preds

def decode_onehot_consensus(onehot) :
    seq = ''
    
    for i in range(onehot.shape[0]) :
        max_j = np.argmax(onehot[i, :])
        
        if max_j == 0 :
            seq += 'A'
        elif max_j == 1 :
            seq += 'C'
        elif max_j == 2 :
            seq += 'G'
        elif max_j == 3 :
            seq += 'T'
    
    return seq

def decode_onehots_consensus(onehots) :
    seqs = [
        decode_onehot_consensus(onehots[i, :, :, 0]) for i in range(onehots.shape[0])
    ]
    
    return seqs
