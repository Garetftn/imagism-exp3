import json, os
from itertools import product
import matplotlib
matplotlib.use('Agg')                      # no windows, just save files
from experiment_3_replication_commented import exp_start

grid = dict(imag_corpus=['orig', 'final'],
            love_corpus=['orig', 'new', 'final'],
            base_line=['computed','three'],
            e_formula=['orig', 'paper'])

# skip runs already in the log, so you can stop and restart
done = set()
if os.path.exists('results/exp3_runs.jsonl'):
    with open('results/exp3_runs.jsonl', encoding='utf-8') as f:
        done = {json.loads(l)['run_id'] for l in f}

for combo in product(*grid.values()):
    cfg = dict(zip(grid, combo))
    run_id = 'imag_{imag_corpus}_love_{love_corpus}_base_{base_line}_entropy_{e_formula}'.format(**cfg)
    if run_id in done:
        print('skip', run_id); continue
    print('run ', run_id)
    try:
        exp_start(**cfg)
    except Exception as e:
        print('FAILED', run_id, repr(e))    # one bad run doesn't stop the grid
        with open('results/exp3_errors.log', 'a') as f:
            f.write(f'{run_id}\t{e!r}\n')

print('ALL DONE')
print('records in /results/exp3_runs.jsonl')