import os
import sys

data = '/Users/jpaggi/engaging/data/rcmc'

regions = {
    'klf1': 'chr8:84,900,000-85,800,000',
    'ppm1g': 'chr5:31,300,000-32,300,000',
    'nanog': 'chr6:122,480,000-122,850,000',
    'fbn2': 'chr18:58,042,000-59,024,000',
    'sox2': 'chr3:33,800,000-35,700,000',
}

capture_probes = f'{data}/captureprobes_mm39.bed'

outdir = sys.argv[1]

if outdir == 'WT_merged':
    cool = f'{data}/WT_minus_inward.mcool'
elif outdir == 'WT_BR1':
    cool = f'{data}/WT_BR1/select_corrected_minus_inward.mcool'
elif outdir == 'WT_BR1_with_dups':
    cool = f'{data}/WT_BR1/select_corrected_minus_inward.dups.mcool'
elif outdir == 'WT_BR2':
    cool = f'{data}/WT_BR2/WT_BR2_minus_inward.mcool'
elif outdir == 'triptolide4h':
    cool = f'{data}/triptolide4h/triptolide4h_minus_inward.mcool'
elif outdir == 'rad21_dmso':
    cool = f'{data}/rad21_dmso/rad21_dmso_minus_inward.mcool'
elif outdir == 'RAD21_BR1':
    cool = f'{data}/RAD21_BR1/select_corrected_minus_inward.mcool'
else:
    raise ValueError('Unknown command')

if not os.path.exists(outdir):
    print(f'Creating directory {outdir}')
    os.mkdir(outdir)

for name, region in regions.items():
    if not os.path.exists(f'{outdir}/{name}.npz'):
        cmd = f"neighbor-balance region-balance {outdir}/{name}.npz {cool} '{region}' --capture-probes-path {capture_probes}"
        print(cmd)
        os.system(cmd)
