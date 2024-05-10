####################################################### linear transformer ####################################################### 
python main.py --suffix=standard --linear -v # standard
python main.py --suffix=1layer -l=1 --linear -v # 1 layer auto
python main.py --suffix=2layers -l=2 --linear -v # 2 layers auto
python main.py --suffix=4layers -l=4 --linear -v # 4 layers auto
python main.py --suffix=sequential --mode=sequential --linear -v # standard sequential
python main.py --suffix=sequential_2layers -l=2 --mode=sequential --linear -v # 2 layer sequential
python main.py --suffix=sequential_4layers -l=4 --mode=sequential --linear -v # 4 layer sequential
python main.py --suffix=large -d=8 -s=20 -n=60 --linear -v # large scale

####################################################### non-linear transformer #######################################################
python main.py --suffix=standard -v # standard
python main.py --suffix=representable --sample_weight -v # representable value function
python main.py --suffix=sequential --mode=sequential -v # standard sequential
python main.py --suffix=large -d=8 -s=20 -n=60 -v # large scale
python main.py --suffix=relu --activation=relu -v # relu activation