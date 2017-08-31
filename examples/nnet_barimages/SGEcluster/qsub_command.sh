mkdir 1ex
mkdir 2ex
mkdir 10ex
mkdir 100ex
mkdir output
mkdir error

qsub -t 1-400 -tc 100 -N barmultiM submit_multiM.sh
