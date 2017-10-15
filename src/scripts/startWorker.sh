#pass as first arg the # of workers to start
for ((c=1;c<=$1;c++))
do
	python -m gensim.models.lda_worker &
done
