python -m cProfile -o profout LDA.py --name=test --data=/home/jonathan/dev/CDS/data/Text/ --modeltype=LDA --index=testPivots.txt > prof.out
python profstats.py profout
