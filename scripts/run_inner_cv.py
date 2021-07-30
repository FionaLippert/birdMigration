import argparse
import os.path as osp
import os, shutil

parser = argparse.ArgumentParser()
parser.add_argument('hp_file', type=str, help='.txt file containing hyperparameter settings')
parser.add_argument('model', type=str, help='model name')
parser.add_argument('year', type=int, help='year to use for validation')
args = parser.parse_args()

home = osp.expanduser('~')
checkpoint_dir = osp.join(home, 'birdMigration', 'checkpoints',
						  f'nested_cv_{args.model}', f'test_{year}', 'hp_grid_search')
os.makedirs(checkpoint_dir, exist_ok=True)
shutil.copy(args.hp_file, checkpoint_dir)


# run array jobs for all hyperparameter settings
with open(args.hp_file, 'r') as f:
	settings = f.readlines()
for s in settings:

srun python run_2.py root=$TMPDIR +sub_dir=setting_$SLURM_ARRAY_TASK_ID \
	action=cv \
	datasource.test_year=$TEST_YEAR \
	job_id=$SLURM_ARRAY_TASK_ID \
	model=$MODEL \
	output_dir=$CHECKPOINTDIR \
	$(head -$SLURM_ARRAY_TASK_ID $HPARAM_FILE | tail -1)

# after all jobs are done, find setting with best average validation error
python determine_best_hp.py $CHECKPOINTDIR best_hp_settings.txt


