"""Convert the official MOSES CSV to the format consumed by train/train.py."""
import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from pathlib import Path
from itertools import islice
from tqdm import tqdm


def process_batch(batch):
    from rdkit import Chem
    from rdkit.Chem import Crippen, QED
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Contrib.SA_Score import sascorer

    results = []
    for line, smiles, split in batch:
        try:
            if split not in {'train', 'test', 'test_scaffolds'}:
                raise ValueError(f'Unknown split: {split!r}')
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                raise ValueError('Invalid or empty molecule')
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            properties = (QED.qed(mol), Crippen.MolLogP(mol), sascorer.calculateScore(mol))
            if not all(math.isfinite(value) for value in properties):
                raise ValueError('Non-finite molecular property')
        except (ValueError, RuntimeError, ZeroDivisionError) as exc:
            results.append((None, [line, smiles, str(exc)]))
        else:
            # Empty scaffolds are intentionally retained.
            results.append(([smiles, split, *properties, scaffold], None))
    return results


def parallel_results(rows, workers, batch_size):
    """Bound pending batches to avoid loading the full dataset into memory."""
    if workers == 1:
        while batch := list(islice(rows, batch_size)):
            yield from process_batch(batch)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        for _ in range(workers * 2):
            batch = list(islice(rows, batch_size))
            if not batch:
                break
            pending.append(executor.submit(process_batch, batch))
        while pending:
            yield from pending.popleft().result()
            batch = list(islice(rows, batch_size))
            if batch:
                pending.append(executor.submit(process_batch, batch))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('datasets/dataset_v1.csv'))
    parser.add_argument('--output', type=Path, default=Path('datasets/moses2.csv'))
    parser.add_argument('--limit', type=int, help='Process only the first N rows for a smoke test')
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1),
                        help='Worker processes (default: up to 4; use 1 for serial processing)')
    parser.add_argument('--batch-size', type=int, default=500)
    args = parser.parse_args()
    if args.workers <= 0 or args.batch_size <= 0:
        parser.error('--workers and --batch-size must be positive')
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.input.resolve() == args.output.resolve():
        parser.error('Input and output must be different files')
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}; choose another --output')
        
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = args.output.with_suffix('.rejected.csv')
    if rejected_path.exists():
        parser.error(f'Rejection report already exists: {rejected_path}')
    # Write to a temporary file so interruption never leaves a completed-looking CSV.
    temporary = args.output.with_suffix('.csv.partial')
    # Count CSV records without loading the dataset into memory.
    with args.input.open(newline='') as source:
        records = csv.reader(source)
        next(records, None)
        total = sum(1 for _ in islice(records, args.limit))
    seen = kept = rejected = 0
    try:
        with args.input.open(newline='') as source:
            reader = csv.DictReader(source)
            columns = {name.strip().lower(): name for name in (reader.fieldnames or [])}
            if not {'smiles', 'split'} <= columns.keys():
                parser.error('Input must contain SMILES and SPLIT columns')
            with temporary.open('x', newline='') as destination, rejected_path.open('x', newline='') as errors:
                writer = csv.writer(destination)
                writer.writerow(['smiles', 'split', 'qed', 'logp', 'sas', 'scaffold_smiles'])
                error_writer = csv.writer(errors)
                error_writer.writerow(['csv_line', 'smiles', 'reason'])
                rows = ((line, (row[columns['smiles']] or '').strip(),
                         (row[columns['split']] or '').strip())
                        for line, row in enumerate(islice(reader, args.limit), start=2))
                results = parallel_results(rows, args.workers, args.batch_size)
                with tqdm(total=total, desc='Processing molecules', unit='mol',
                          dynamic_ncols=True) as progress:
                    for output, error in results:
                        seen += 1
                        if error is not None:
                            rejected += 1
                            error_writer.writerow(error)
                        else:
                            kept += 1
                            writer.writerow(output)
                        progress.update(1)
                        if seen % 100 == 0 or seen == total:
                            progress.set_postfix(kept=kept, rejected=rejected)
        if not kept:
            raise ValueError('No usable molecules; no final output written')
        temporary.rename(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f'Done: processed={seen:,}, kept={kept:,}, rejected={rejected:,}')
    print(f'Output: {args.output}\nRejection report: {rejected_path}')


if __name__ == '__main__':
    main()


'''
python scripts/preprocess_moses.py
python scripts/preprocess_moses.py --limit 100 --output datasets/moses_sample.csv
'''
