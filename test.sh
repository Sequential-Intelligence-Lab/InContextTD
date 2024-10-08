python demo.py --n_mrps=50 --save_dir=logs/test
python verify.py --num_trials=5 --save_dir=logs/test
python main.py --n_mrps=20 -v --seed 1 2 3 --save_dir=logs/test/linear
python main.py --n_mrps=20 --activation=softmax -v --seed 1 2 3 --save_dir=logs/test/nonlinear