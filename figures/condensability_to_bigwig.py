import pandas as pd
import numpy as np
import pyBigWig

def read_condensibility(cell_type='GM12878'):
    root = '/home/joepaggi/orcd/pool/condensibility'
    if cell_type == 'GM12878':
         rep1 = f'{root}/GSE252941_GM_NCP_sp_1rep_deep_1kb_score.gtab.txt'
         rep2 = f'{root}/GSE252941_GM_NCP_sp_2rep_deep_1kb_score.gtab.txt'
    elif cell_type == 'H1':
        rep1 = f'{root}/GSE252941_H1_NCP_sp_1rep_deep_1kb_score.gtab.txt'
        rep2 = f'{root}/GSE252941_H1_NCP_sp_2rep_deep_1kb_score.gtab.txt'
    else:
        raise ValueError('Cell type not recognized')
    condensibility1 = pd.read_csv(rep1, sep='\t')
    condensibility2 = pd.read_csv(rep2, sep='\t')
    condensibility1 = condensibility1.set_index(['Chromosome', 'Start', 'End'])
    condensibility2 = condensibility2.set_index(['Chromosome', 'Start', 'End'])
    condensibility = condensibility1.join(condensibility2, how='outer')
    condensibility = condensibility.reset_index()

    if cell_type == 'GM12878':
        condensibility = condensibility.rename(columns={
            'GM_NCP_sp_8_1rep_deep': '8_rep1',
            'GM_NCP_sp_8_2rep_deep': '8_rep2',
        })
    elif cell_type == 'H1':
        condensibility = condensibility.rename(columns={
            'H1_NCP_sp_8_1rep_deep': '8_rep1',
            'H1_NCP_sp_8_2rep_deep': '8_rep2',
        })
    else:
        raise ValueError('Cell type not recognized')
    return condensibility


def get_condensibility_track(condensibility, chrom):
    con = condensibility[condensibility.Chromosome == chrom]
    val = (con['8_rep1'] + con['8_rep2']) / 2
    return con['Start'].to_numpy(), con['End'].to_numpy(),  val.to_numpy()

def write_condensibility_bigwig(cell_type='GM12878'):
    condensibility = read_condensibility(cell_type=cell_type)
    bw_path = f'condensibility_{cell_type}.bw'
    bw = pyBigWig.open(bw_path, 'w')
    chroms = []
    lengths = []
    for chrom in condensibility['Chromosome'].unique():
        chrom_length = condensibility[condensibility['Chromosome'] == chrom]['End'].max()
        chroms.append(chrom)
        lengths.append(chrom_length)
    bw.addHeader(list(zip(chroms, lengths)))
    for chrom in chroms:
        start, end, val = get_condensibility_track(condensibility, chrom)
        bw.addEntries([chrom]*len(val), start, ends=end, values=val.tolist())
    bw.close()

if __name__ == '__main__':
    write_condensibility_bigwig(cell_type='GM12878')
    write_condensibility_bigwig(cell_type='H1')
