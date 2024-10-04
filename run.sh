####################################################### linear transformer ####################################################### 
python main.py --suffix=linear_standard -l=3   # standard

python main.py --suffix=linear_2layers -l=2   # 2 layers auto

python main.py --suffix=linear_4layers -l=4   # 4 layers auto

python main.py --suffix=linear_seq --mode=sequential  # standard sequential

python main.py --suffix=linear_seq_2layers -l=2 --mode=sequential   # 2 layer sequential

python main.py --suffix=linear_seq_4layers -l=4 --mode=sequential  # 4 layer sequential

python main.py --suffix=linear_large -d=8 -s=20 -n=60 # larger scale

####################################################### non-linear transformer #######################################################
python main.py --suffix=nonlinear_standard --activation=softmax # standard

python main.py --suffix=nonlinear_rep --activation=softmax --representable # representable value function