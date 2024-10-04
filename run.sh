####################################################### linear transformer ####################################################### 
python main.py --suffix=linear_standard -l=3 --linear   # standard

python main.py --suffix=linear_2layers -l=2 --linear  # 2 layers auto

python main.py --suffix=linear_4layers -l=4 --linear  # 4 layers auto

python main.py --suffix=linear_sequential --mode=sequential --linear # standard sequential

python main.py --suffix=linear_sequential_2layers -l=2 --mode=sequential --linear  # 2 layer sequential

python main.py --suffix=linear_sequential_4layers -l=4 --mode=sequential --linear # 4 layer sequential

python main.py --suffix=linear_large -d=8 -s=20 -n=60 --linear # larger scale

####################################################### non-linear transformer #######################################################
python main.py --suffix=nonlinear_standard --activation=softmax # standard

python main.py --suffix=nonlinear_representable --activation=softmax --sample_weight # representable value function

wait # wait for all background jobs to finish