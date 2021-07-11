#!/usr/bin/env python3
# coding: utf-8


import os.path
import sys
import copy

import argparse

import numpy as np
import pandas as pd
import scipy.stats as ss

import json

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import seaborn as sns
except ImportError:
    sns = None

from itertools import product

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import DNAAlphabet


try:
    import bz2
except ImportError:
    bz2 = None

try:
    import gzip
except ImportError:
    gzip = None

try:
    import lzma
except ImportError:
    lzma = None


def openfile(filename, mode="rt", *args, expanduser=False, expandvars=False,
             makedirs=False, **kwargs):
    """
    Open filename and return a corresponding file object.
    Reference: https://github.com/luismsgomes/openfile
    """
    if filename == "-" or filename is None:
        return sys.stdin if "r" in mode else sys.stdout
    if expanduser:
        filename = os.path.expanduser(filename)
    if expandvars:
        filename = os.path.expandvars(filename)
    if makedirs and ("a" in mode or "w" in mode):
        parentdir = os.path.dirname(filename)
        if not os.path.isdir(parentdir):
            os.makedirs(parentdir)
    if filename.endswith(".gz"):
        if gzip is None:
            raise NotImplementedError
        _open = gzip.open
    elif filename.endswith(".bz2"):
        if bz2 is None:
            raise NotImplementedError
        _open = bz2.open
    elif filename.endswith(".xz") or filename.endswith(".lzma"):
        if lzma is None:
            raise NotImplementedError
        _open = lzma.open
    else:
        _open = open
    return _open(filename, mode, *args, **kwargs)


####################
# CONSTANTS
####################


COMP = {"A": "T",
        "C": "G",
        "G": "C",
        "T": "A"}
CODE = {"A": 0,
        "C": 1,
        "G": 2,
        "T": 3}
UNCODE = {v: k for k, v in CODE.items()}


####################
# GETTING SEQ-S
####################


def calc_cpm(x):
    return np.log10(x / x.sum()) + 6


def get_nr(fastq_file, sample=0, verbose=False):
    with openfile(fastq_file, "rt") as handle:
        parser = SeqIO.parse(handle, "fastq", DNAAlphabet())
        nonunique = pd.Series([str(rec.seq) for rec in parser if "N" not in rec.seq])
        if sample > 0 and sample < nonunique.shape[0]:
            nonunique = nonunique.sample(n=sample, replace=False)
        cpm = calc_cpm(nonunique.value_counts().sort_index())
        unique = pd.Series(cpm.index)
        
    if verbose:
        sys.stderr.write(f"Extracted {len(unique)} non-redundant sequences; {len(nonunique)} non-unique sequences\n")
    return cpm, nonunique, unique


####################
# SHUFFLING SEQ-S
####################


def shuffle_string(s):
    chars = list(s)
    np.random.shuffle(chars)
    return "".join(chars)


def compute_count(s):
    # P. Clote, Oct 2003
    # Initialize lists and mono- and dinucleotide dictionaries
    dct = {i: list() for i in CODE}
    nucl_list = list(CODE.keys())
    nucl_cnt = dict()
    dinucl_cnt  = dict()
    for x in nucl_list:
        nucl_cnt[x]=0
        dinucl_cnt[x]={}
        for y in nucl_list:
            dinucl_cnt[x][y]=0
    
    nucl_cnt[s[0]] = 1
    nucl_total = 1
    dinucl_total = 0
    for i in range(len(s) - 1):
        x = s[i]
        y = s[i + 1]
        dct[x].append(y)
        nucl_cnt[y] += 1
        nucl_total += 1
        dinucl_cnt[x][y] += 1
        dinucl_total += 1
    assert nucl_total == len(s)
    assert dinucl_total == len(s) - 1
    return dinucl_cnt, dct


def choose_edge(x, dinucl_cnt):
    # P. Clote, Oct 2003
    z = np.random.random()
    denom = dinucl_cnt[x]['A'] + dinucl_cnt[x]['C'] + dinucl_cnt[x]['G'] + dinucl_cnt[x]['T']
    numerator = dinucl_cnt[x]['A']
    if z < numerator / denom:
        dinucl_cnt[x]['A'] -= 1
        return 'A'
    numerator += dinucl_cnt[x]['C']
    if z < numerator / denom:
        dinucl_cnt[x]['C'] -= 1
        return 'C'
    numerator += dinucl_cnt[x]['G']
    if z < numerator / denom:
        dinucl_cnt[x]['G'] -= 1
        return 'G'
    dinucl_cnt[x]['T'] -= 1
    return 'T'


def connected_to_last(edge_list, nucl_list, last_char):
    # P. Clote, Oct 2003
    dct = {x: 0 for x in nucl_list}
    for edge in edge_list:
        a = edge[0]
        b = edge[1]
        if b == last_char:
            dct[a] = 1
    for i in range(2):
        for edge in edge_list:
            a = edge[0]
            b = edge[1]
            if dct[b] == 1:
                dct[a] = 1
    for x in nucl_list:
        if x != last_char and dct[x] == 0:
            return False
    return True
 

def eulerian(s):
    # P. Clote, Oct 2003
    dinucl_cnt, dct = compute_count(s)
    nucl_list = []
    for x in CODE.keys():
        if x in s:
            nucl_list.append(x)

    last_char = s[-1]
    edge_list = []
    for x in nucl_list:
        if x != last_char:
            edge_list.append((x, choose_edge(x, dinucl_cnt)))
    ok = connected_to_last(edge_list, nucl_list, last_char)
    return ok, edge_list, nucl_list


def shuffle_edge_list(lst):
    # P. Clote, Oct 2003
    n = len(lst)
    barrier = n
    for i in range(n - 1):
        z = int(np.random.random() * barrier)
        lst[z], lst[barrier - 1] = lst[barrier - 1], lst[z]
        barrier -= 1
    return lst


def shuffle_string_dinucl(s):
    # P. Clote, Oct 2003
    ok = False
    while not ok:
        ok, edge_list, nucl_list = eulerian(s)
    dinucl_cnt, dct = compute_count(s)

    #remove last edges from each vertex list, shuffle, then add back
    #the removed edges at end of vertex lists.
    for x, y in edge_list:
        dct[x].remove(y)
    for x in nucl_list:
        shuffle_edge_list(dct[x])
    for x, y in edge_list:
        dct[x].append(y)
    
    lst = [s[0]]
    prev_char = s[0]
    for i in range(len(s)-2):
        char = dct[prev_char][0] 
        lst.append(char)
        del dct[prev_char][0]
        prev_char = char
    lst.append(s[-1])
    return "".join(lst)


####################
# SEQ-S OPERATIONS
####################


def revcomp(seq):
    return "".join((COMP[x] for x in seq[::-1]))


def get_all_kmers(k, alphabet="ACGT"):
    kdict = list()
    for kmer in product(alphabet, repeat=k):
        kmer = "".join(kmer)
        kdict.append(kmer)
    return kdict


def get_rc_kmers(k, alphabet="ACGT"):
    kdict = dict()
    for kmer in product(alphabet, repeat=k):
        kmer = "".join(kmer)
        rc = revcomp(kmer)
        if rc in kdict:
            kdict[kmer] = False
        else:
            kdict[kmer] = True
    return kdict


def invert_kmers(data, kdict, df=True):
    if df:
        new_data = pd.DataFrame(dtype="float64")
    else:
        new_data = pd.Series(dtype="float64")
    for kmer in kdict:
        if kdict[kmer]:
            rc = revcomp(kmer)
            new_data[kmer] = data[kmer] + data[rc]
            new_data[rc] = new_data[kmer] 
    return new_data


def encode(kmer):
    k = len(kmer)
    code = 0
    for i in range(0, k, 1):
        itercode = CODE[kmer[i]]
        code = (code << 2) + itercode
    return code


def uncode(code, k=4):
    kmer = []
    for i in range(k - 1, -1, -1):
        ex = 4 ** i
        code, itercode = code % ex, code // ex
        kmer.append(UNCODE[itercode])
    assert code == 0, "Wrong code!"
    return "".join(kmer)


####################
# K-MER COUNTING
####################


def make_kmer_vect(unique_seqs, k=4, return_kmer_df=False):
    letters = np.array(unique_seqs.apply(lambda x: [CODE[sym] for sym in x]).tolist())
    cp = letters.copy()
    for roll in range(1, k):
        letters = (letters << 2) + np.roll(cp, -roll, axis=1)
    
    letters = pd.DataFrame(letters[:,:-k + 1], index=unique_seqs)
    mx = letters.melt().groupby(by="value").count().squeeze()
    mx.rename(lambda x: uncode(x, k), inplace=True)
    if return_kmer_df:
        return mx, letters
    else:
        return mx


def calc_kmer_scores(kmer_code_df, kmer_scores, verbose=False):
    # attention: `kmer_code_df` is mutated!
    kmer_code_df.rename_axis("seq", axis=0, inplace=True)
    kmer_code_df.reset_index(inplace=True)
    kmer_scores = kmer_scores.rename(index=encode)
    
    codes = kmer_code_df.melt(id_vars=["seq"], var_name="position", value_name="kmer")
    codes.drop(labels=["position"], axis=1, inplace=True)
    codes["kmer"] = codes["kmer"].map(kmer_scores)
    scores = codes.groupby(by=["seq"])["kmer"].sum()
    
    return scores


def calc_kmers(non_unique_seqs, unique_seqs, k=4, dinucl=False, invert=True, verbose=False):
    if verbose:
        sys.stderr.write(f"Started analyzing {len(unique_seqs)} sequences\n")

    if invert:
        kdict = get_rc_kmers(k)
        kmers_list = sorted(kdict.keys())
    else:
        kmers_list = sorted(get_all_kmers(k))

    if verbose:
        assert len(kmers_list) == 4 ** k, f"Wrong number of {k}-mers generated"
        sys.stderr.write(f"Generated {len(kmers_list)} {k}-mers\n")

    experiment = unique_seqs

    if dinucl:
        control = unique_seqs.apply(shuffle_string_dinucl)
    else:
        control = unique_seqs.apply(shuffle_string)
    if verbose:
        sys.stderr.write("Control generated\n")
    
    m0 = make_kmer_vect(control, k=k, return_kmer_df=False).reindex(kmers_list, fill_value=0)
    if verbose:
        sys.stderr.write(f"Counted {k}-mers for control\n")
    if invert:
        m0 = invert_kmers(m0, kdict, df=False)
    
    m1, exp_kmer_df = make_kmer_vect(experiment, k=k, return_kmer_df=True)
    m1 = m1.reindex(kmers_list, fill_value=0)
    if verbose:
        sys.stderr.write(f"Counted {k}-mers for experiment\n")
    if invert:
        m1 = invert_kmers(m1, kdict, df=False)

    m0 /= m0.sum()
    m1 /= m1.sum()

    c1 = np.log10((m1 / m0).fillna(1))
    x1 = calc_kmer_scores(exp_kmer_df, c1)

    if verbose:
        sys.stderr.write(f"Scored sequences\n")

    return x1, c1


####################
# FILTERING RESULTS
####################


def quantile_slice(alpha, beta, cpm, scores, verbose=False):
    scores_enr, scores_etc = split_by_alpha(alpha, cpm, scores, verbose=verbose)
    choice = split_by_beta(beta, scores_enr, scores_etc, verbose=verbose)
    calc_statistics(choice, scores_enr, scores_etc, verbose=verbose)
    return scores_enr, scores_etc, choice


def split_by_alpha(alpha, cpm, scores, verbose=False):
    if not (scores.index == cpm.index).all(axis=None):
        raise RuntimeError("Scores and CPM have different indices")
    singletons = (cpm == cpm.min())
    if alpha == 0:
        cpm_q = cpm.min()
        scores_enr = scores[cpm >= cpm_q]
        scores_etc = scores[cpm <= cpm_q]  # singletons are used as background distribution
    elif 0 < alpha < 1:
        cpm_q = cpm.quantile(alpha)
        scores_enr = scores[cpm > cpm_q]  # singletons are excluded, because cpm_q cannot be less than a minimal cpm
        scores_etc = scores[cpm <= cpm_q]
    elif alpha >= 1:
        alpha = int(alpha)
        top_cpm = cpm.sample(frac=1.0, replace=False).sort_values(ascending=False).iloc[:alpha]
        cpm_q = top_cpm
        top_cpm_slice = scores.index.isin(top_cpm.index) & (~singletons)
        scores_enr = scores[top_cpm_slice]
        scores_etc = scores[~top_cpm_slice]
    else:
        raise ValueError("Alpha value must be either a quantile [0; 1) or a number of reads from top [1; +inf)")
    if verbose:
        sys.stderr.write(f"Alpha = {alpha}; chosen top {scores_enr.shape[0]} sequences by CPM\n")
    return scores_enr, scores_etc


def split_by_beta(beta, scores_enr, scores_etc, verbose=False):
    if beta == 0:
        choice = scores_enr
    elif 0 < beta < 1:
        scores_q = scores_etc.quantile(beta)
        choice = scores_enr[scores_enr > scores_q]
    elif beta >= 1:
        beta = int(beta)
        choice = scores_enr.sample(frac=1.0, replace=False).sort_values(ascending=False).iloc[:beta]
    else:
        raise ValueError("Beta value must be either a quantile [0; 1) or a number of reads from top [1; +inf)")
    if verbose:
        sys.stderr.write(f"Beta = {beta}; chosen top {choice.shape[0]} sequences by k-mer score\n")
    return choice


####################
# STATS AND CHARTS
####################


def calc_statistics(choice, scores_enr, scores_etc, fc=True, ks=True, u=True, medians=True, verbose=False):
    out = dict()
    if not verbose:
        return out
    if fc:
        new_q = choice.shape[0] / scores_enr.shape[0]
        min_score = choice.min()
        ref_n = (scores_etc >= min_score).sum()
        ref_q = ref_n / scores_etc.shape[0]
        fc_val = new_q / ref_q
        if verbose:
            sys.stderr.write(f"Quantile FC = {fc_val}\n")
        out["fc"] = fc_val
    if ks:
        ks_test = ss.ks_2samp(scores_etc, scores_enr, alternative="greater")
        if verbose:
            sys.stderr.write(f"KS = {ks_test.statistic}; KS_p = {ks_test.pvalue}\n")
        out["ks"] = ks_test.pvalue
    if u:
        u_test = ss.mannwhitneyu(scores_etc, scores_enr, use_continuity=True, alternative="less")
        if verbose:
            sys.stderr.write(f"U = {u_test.statistic}; U_p = {u_test.pvalue}\n")
        out["u"] = u_test.pvalue
    if medians:
        med_val = scores_enr.median() - scores_etc.median()
        if verbose:
            sys.stderr.write(f"Median diff = {med_val}\n")
        out["medians"] = med_val
    return out


def trace_chart(cpm, scores, scores_enr, scores_etc, path, title, thres_score=None, k=4, verbose=False):
    # Charting points
    chart = sns.jointplot(cpm, scores, height=10, color="k", alpha=0.33, marker=".", marginal_kws=dict(norm_hist=True, bins=50))
    
    # Charting thresholds
    if thres_score is not None:
        colors = sns.color_palette()
        chart.ax_joint.cla()
        cpm_l, scores_l = cpm.reindex(scores_etc.index), scores_etc
        cpm_r, scores_r = cpm.reindex(scores_enr.index), scores_enr
        chart.ax_joint.scatter(cpm_l, scores_l, color=colors[0], alpha=0.33, marker=".")
        chart.ax_joint.scatter(cpm_r, scores_r, color=colors[1], alpha=0.33, marker=".")
        chart.ax_joint.axhline(thres_score, color="k", linestyle="--", lw=1)
        chart.ax_marg_y.cla()
        chart.ax_marg_y.hist(scores_l, density=True, color=colors[0], alpha=0.33, bins=50, orientation="horizontal")
        chart.ax_marg_y.hist(scores_r, density=True, color=colors[1], alpha=0.33, bins=50, orientation="horizontal")
        chart.ax_marg_y.axhline(thres_score, color="k", linestyle="--", lw=1)
        chart.ax_marg_y.set_xticks([])
        chart.ax_marg_y.tick_params(axis="x", which="both", top=False, bottom=False, labeltop=False, labelbottom=False)
        chart.ax_marg_y.tick_params(axis="y", which="both", labelleft=False, labelright=False)
    
    # Adding labels
    chart.set_axis_labels("Sequence log10 CPM", f"{k}-mer enrichment score")
    if title is not None:
        chart.fig.suptitle(f"{title}".replace("&", "\n"))
    
    # Saving figure
    chart.fig.tight_layout(pad=1.0)
    chart.savefig(path, bbox_inches="tight")
    plt.close()
    if verbose:
        sys.stderr.write(f"Traced enrichment-relevance chart\n")
    return chart


####################
# WRITING RESULTS
####################


def get_f_flanks(f=0, left_flank="", right_flank=""):
    if f == 0:
        return "", ""
    if f == -1:
        return left_flank, right_flank
    if f < 0:
        raise ValueError("Flank length cannot be negative")
    left_flank = left_flank[-f:].rjust(f, "N")
    right_flank = right_flank[:f].ljust(f, "N")
    return left_flank, right_flank


def write_top_seqs(scores, outfile_name, n_flank=0, f_flank=0, left_flank="", right_flank="", verbose=False):
    top = scores.sort_values(ascending=False)
    to_write = list()
    n_flank = "N" * n_flank
    left_flank, right_flank = get_f_flanks(f_flank, left_flank, right_flank)
    for seq, score in top.iteritems():
        record = SeqRecord(Seq(f"{n_flank}{left_flank}{seq}{right_flank}{n_flank}"), id=f" {score:.4f}", description="")
        to_write.append(record)
    with openfile(outfile_name, "wt") as outfile:
        SeqIO.write(to_write, outfile, "fasta")
    
    if verbose:
        sys.stderr.write(f"Best {top.shape[0]} sequences written\n")


####################
# AGGREGATED FUNC
####################


def process(fastq, out, dinucl=False, sample=0, k=4,
            alpha=0.95, beta=0.99, n_flank=0, f_flank=0,
            left_flank="", right_flank="", invert=True,
            chart, chart_title, verbose=False):
    if chart is not None:
        if plt is None:
            raise ImportError("Module 'matplotlib' failed to import")
        elif sns is None:
            raise ImportError("Module 'seaborn' failed to import")
    cpm, allseqs, unique = get_nr(fastq, sample=sample, verbose=verbose)
    scores, kmers = calc_kmers(allseqs, unique, k=k, dinucl=dinucl, invert=invert, verbose=verbose)
    scores_enr, scores_etc, scores_top = quantile_slice(alpha, beta, cpm, scores, verbose)
    if chart is not None:
        thres_score = scores_top.min()
        ch = trace_chart(cpm, scores, scores_enr, scores_etc, chart, chart_title, thres_score, k=k, verbose=verbose)
    write_top_seqs(scores_top, out, n_flank, f_flank, left_flank, right_flank, verbose=verbose)
    
    if verbose:
        sys.stderr.write("Done!\n\n")


####################
# LAUNCHING SCRIPT
####################


if __name__ == "__main__":
    if sns is not None:
        sns.set()
        sns.set_style("whitegrid")
    
    description_text = """Finds top N unique sequences by modified P.Bucher's k-mers algorithm.
The script does NOT support ambiguous IUPAC codes (ATGC only)."""
    
    example_text = """examples:
  cat selex_cycle4.fastq.gz | extract_topk.py -v > kmer_ext.fasta 2> k-mer_ext.log
  extract_topk.py --fastq selex_cycle4.fastq --out kmer_ext.fasta
  extract_topk.py --fastq selex_cycle4.fastq --out kmer_ext.fasta -k 5 --chart selex_cycle4_5mers.png --title "Cycle 4 5-mer enrichment"
  extract_topk.py --fastq selex_cycle4.fastq.gz --out lo-fi_kmer_ext.fasta -a 0.8 -b 0.8
  extract_topk.py --fastq selex_cycle4.fastq.gz --out flanked_kmer_ext.fasta -n 5 -f 3 --prefix TGACTA --suffix AAGATC
"""
    
    parser = argparse.ArgumentParser(description=description_text,
                                     epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("--fastq", default="-", metavar="fastq", help="input fastq file (default: stdin)")
    parser.add_argument("--out", default="-", metavar="fasta", help="output fasta file (default: stdout)")
    parser.add_argument("-d", "--dinucl", help="use dinucleotide shuffling instead of mononucleotide", action="store_true")
    parser.add_argument("--sample", default=0, metavar="N", type=int, help="number of sequence subset to analyze (default: 0 [no sampling])")

    parser.add_argument("-k", default=4, metavar="K", type=int, help="length of a k-mer (default: 4)")
    parser.add_argument("-a", "--alpha", default=100000, metavar="P", type=float, help="quantile level of 'baseline' sequences [0 to 1) or the number of "\
                        "non-baseline sequences (1, 2, ...) (default: 100'000). 'Single-hit' sequences are always excluded.")
    parser.add_argument("-b", "--beta", default=0.99, metavar="P", type=float, help="threshold 'baseline' level for 'non-baseline' [0 to 1) or the number "\
                        "of non-baseline sequences (1, 2, ...) (default: 0.99)")

    parser.add_argument("-n", default=0, metavar="N", type=int, help="length of N-flank to be added (default: 0)")
    parser.add_argument("-f", default=-1, metavar="F", type=int, help="length of non-N-flank to be added. "\
                        "If either prefix or suffix are shorter than F, N-flanks are extended to match the length. "\
                        "To avoid this, use default F (-1), the flanks will be added as-is.")
    parser.add_argument("--prefix", default="", metavar="sequence", help="prefix sequence. "\
                        "Only the last F letters are prepended to the output, unless F is not -1.")
    parser.add_argument("--suffix", default="", metavar="sequence", help="suffix sequence. "\
                        "Only the first F letters are appended to the output, unless F is not -1.")

    parser.add_argument("-i", "--noinvert", help="don't summarize counts of the reverse-complement k-mers in the score", action="store_false")

    parser.add_argument("--chart", default=None, metavar="svg/png/etc", help="trace an enrichment-relevance chart (optional)")
    parser.add_argument("--title", default=None, metavar="TITLE", help="use custom title for the chart (used only when the parameter --chart is specified)")

    parser.add_argument("--seed", default=13, metavar="SEED", type=int, help="seed for RNG (default: 13)")
    parser.add_argument("-v", "--verbose", help="verbose output", action="store_true")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    process(fastq=args.fastq,
            out=args.out,
            dinucl=args.dinucl,
            sample=args.sample,
            chart=args.chart,
            chart_title=args.title,
            k=args.k,
            alpha=args.alpha,
            beta=args.beta,
            n_flank=args.n,
            f_flank=args.f,
            left_flank=args.prefix,
            right_flank=args.suffix,
            invert=args.noinvert,
            verbose=args.verbose)

