####################################################### linear transformer ####################################################### 
python main.py --suffix=standard --linear -v & # standard
sleep 1
python main.py --suffix=2layers -l=2 --linear -v & # 2 layers auto
sleep 1
python main.py --suffix=4layers -l=4 --linear -v & # 4 layers auto
sleep 1
python main.py --suffix=sequential --mode=sequential --linear -v & # standard sequential
sleep 1
python main.py --suffix=sequential_2layers -l=2 --mode=sequential --linear -v &  # 2 layer sequential
sleep 1
python main.py --suffix=sequential_4layers -l=4 --mode=sequential --linear -v & # 4 layer sequential
sleep 1
python main.py --suffix=large -d=8 -s=20 -n=60 --linear -v & # larger scale
sleep 1
wait

####################################################### non-linear transformer #######################################################
python main.py --suffix=standard -v & # standard
sleep 1
python main.py --suffix=representable --sample_weight -v & # representable value function
sleep 1
python main.py --suffix=sequential --mode=sequential  -v & # standard sequential
sleep 1
wait # wait for all background jobs to finish