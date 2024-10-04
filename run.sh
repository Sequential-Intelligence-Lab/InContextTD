####################################################### linear transformer ####################################################### 
python main.py --suffix=linear_standard --linear   # standard
sleep 2
python main.py --suffix=linear_2layers -l=2 --linear  # 2 layers auto
sleep 2
python main.py --suffix=linear_4layers -l=4 --linear  # 4 layers auto
sleep 2
python main.py --suffix=linear_sequential --mode=sequential --linear # standard sequential
sleep 2
python main.py --suffix=linear_sequential_2layers -l=2 --mode=sequential --linear  # 2 layer sequential
sleep 2
python main.py --suffix=linear_sequential_4layers -l=4 --mode=sequential --linear -v # 4 layer sequential
sleep 2
python main.py --suffix=linear_large -d=8 -s=20 -n=60 --linear -v # larger scale
sleep 2
wait

####################################################### non-linear transformer #######################################################
python main.py --suffix=nonlinear_standard --activation=softmax # standard
sleep 2
python main.py --suffix=nonlinear_representable --activation=softmax --sample_weight # representable value function
sleep 2
wait # wait for all background jobs to finish