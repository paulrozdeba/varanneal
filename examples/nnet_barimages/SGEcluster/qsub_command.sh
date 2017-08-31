mkdir -p 1ex
mkdir -p 2ex
mkdir -p 10ex
mkdir -p 100ex
mkdir -p output
mkdir -p error

qsub -t 1-400 -tc 100 -N barmultiM submit_multiM.sh
