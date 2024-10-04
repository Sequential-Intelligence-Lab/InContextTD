####################################################### linear transformer ####################################################### 
python main.py --suffix=linear_standard --linear -v  & # standard
sleep 2
python main.py --suffix=linear_2layers -l=2 --linear -v & # 2 layers auto
sleep 2
python main.py --suffix=linear_4layers -l=4 --linear -v & # 4 layers auto
sleep 2
python main.py --suffix=linear_sequential --mode=sequential --linear -v & # standard sequential
sleep 2
python main.py --suffix=linear_sequential_2layers -l=2 --mode=sequential --linear -v &  # 2 layer sequential
sleep 2
python main.py --suffix=linear_sequential_4layers -l=4 --mode=sequential --linear -v & # 4 layer sequential
sleep 2
python main.py --suffix=linear_large -d=8 -s=20 -n=60 --linear -v & # larger scale
sleep 2
wait

####################################################### non-linear transformer #######################################################
python main.py --suffix=nonlinear_standard -v & # standard
sleep 2
python main.py --suffix=nonlinear_representable --sample_weight -v & # representable value function
sleep 2
wait # wait for all background jobs to finish