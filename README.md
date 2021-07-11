# K-mer motif enrichment

A script for selecting sequences enriched with k-mers from a sequence set.

## Requirements

* Python 3
* NumPy
* Pandas
* SciPy
* Matplotlib (_for charting_)
* Seaborn (_for charting_)
* BioPython

## How to use

Command syntax:

```
extract_topk.py [-h] [--fastq fastq] [--out fasta] [-d] [--sample N] [-k K] [-a P] [-b P] [-n N] [-f F] [--prefix sequence] [--suffix sequence] [-i] [--chart svg/png/etc] [--title TITLE] [--seed SEED] [-v]
```

Finds top unique sequences by their highest k-mer enrichment for further motif discovery.
The script does NOT support ambiguous IUPAC codes (ATGC only).

```
Arguments:
  -h, --help           show this help message and exit
  --fastq fastq        input fastq file (default: stdin)
  --out fasta          output fasta file (default: stdout)
  -d, --dinucl         use dinucleotide shuffling instead of mononucleotide
  --sample N           number of sequence subset to analyze (default: 0 [no sampling])
  -k K                 length of a k-mer (default: 4)
  -a P, --alpha P      quantile level of 'baseline' sequences [0 to 1) or the number
                       of non-baseline sequences (1, 2, ...) (default: 100'000).
                       'Single-hit' sequences are always excluded.
  -b P, --beta P       threshold 'baseline' level for 'non-baseline' [0 to 1) or
                       the number of non-baseline sequences (1, 2, ...) (default: 0.99)
  -n N                 length of N-flank to be added (default: 0)
  -f F                 length of non-N-flank to be added. If either prefix or suffix
                       are shorter than F, N-flanks are extended to match the length.
                       To avoid this, use default F (-1), the flanks will be added as-is.
  --prefix sequence    prefix sequence. Only the last F letters are prepended to
                       the output, unless F is not -1.
  --suffix sequence    suffix sequence. Only the first F letters are appended to
                       the output, unless F is not -1.
  -i, --noinvert       don't summarize counts of the reverse-complement k-mers in the score
  --chart svg/png/etc  trace an enrichment-relevance chart (optional)
  --title TITLE        use custom title for the chart (used only when the parameter
                       --chart is specified)
  --seed SEED          seed for RNG (default: 13)
  -v, --verbose        verbose output
```
### Example usage

```sh
  cat selex_cycle4.fastq.gz | extract_topk.py -v > kmer_ext.fasta 2> k-mer_ext.log
  extract_topk.py --fastq selex_cycle4.fastq --out kmer_ext.fasta
  extract_topk.py --fastq selex_cycle4.fastq --out kmer_ext.fasta -k 5 --chart selex_cycle4_5mers.png --title "Cycle 4 5-mer enrichment"
  extract_topk.py --fastq selex_cycle4.fastq.gz --out lo-fi_kmer_ext.fasta -a 0.8 -b 0.8
  extract_topk.py --fastq selex_cycle4.fastq.gz --out flanked_kmer_ext.fasta -n 5 -f 3 --prefix TGACTA --suffix AAGATC
```


## Citing and authors

This script will probably be used in some article. I'll provide a DOI, should it get published.

The algorithm was initially proposed by Philipp Bucher, and the further modifications and changes were proposed by Ivan Kulakovskii and Arsenii Zinkevich. The final script was written by Arsenii Zinkevich.
