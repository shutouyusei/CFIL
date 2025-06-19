from .rec_trajectory import RecTrajectory
from .save_trajectory import SaveTrajectory
import sys

# example
# python -m record 1-1

args = sys.argv
if args[1] != None:
    obs,action = RecTrajectory().start_rec(args[1])
    save = SaveTrajectory()
    save.save(args[1],obs,action,"../Data")
else:
    print("please input level")
