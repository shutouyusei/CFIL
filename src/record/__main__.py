from .rec_trajectory import RecTrajectory,SaveTrajectory
import sys

args = sys.argv
if args[1] != None:
    rec = RecTrajectory()
    obs,action = rec.start_rec(args[1])
    save = SaveTrajectory()
    save.save(args[1],obs,action)
else:
    print("please input level")
