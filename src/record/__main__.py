from .rec_trajectory import RecTrajectory
from .save_trajectory import SaveTrajectory
import sys

# example
# python -m record 1-1

args = sys.argv
if args[1] != None:
    obs,action = RecTrajectory().start_rec(args[1])
    save = SaveTrajectory()
    print("Is the game successful ? [y/n]")
    while True :
        is_success = input()
        if is_success == 'y':
            save.save(args[1],obs,action,"../Data/success")
            break
        elif is_success == 'n':
            save.save(args[1],obs,action,"../Data/failure")
            break
        else:
            print("please input y or n")
            continue
else:
    print("please input level")
