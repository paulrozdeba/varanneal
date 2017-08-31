#!/share/apps/opt/python/2.7.9/bin/python2
#$ -S /share/apps/opt/python/2.7.9/bin/python2
#$ -V
#$ -cwd
#$ -j y
#$ -M youremail@domain.com
#$ -o ./output
#$ -e ./error
#$ -q batch.q

import os

M = [1, 2, 10, 100]  # M values to loop through
Ninit = 100  # Number of initializations for annealing

# Here we fetch the SGE task ID, which is important for deciding on which 
# M value to use, which initialization, etc.
SGE_TASK_ID = int(os.getenv("SGE_TASK_ID", 0))

i_M = (int(SGE_TASK_ID - 1) / int(Ninit)) % int(len(M))
initID = int(SGE_TASK_ID - 1) % Ninit + 1
adolcID = SGE_TASK_ID % 2000

print("M = %d"%(M[i_M],))
print("initID = %d"%(initID,))
print("SGE_TASK_ID = %d"%(SGE_TASK_ID,))

print(os.system("uname -n"))

os.system("python2 SGE_bardata_anneal.py %d %d %d"%(initID, M[i_M], adolcID))
