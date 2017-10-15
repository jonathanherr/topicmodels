export PYRO_LOGLEVEL=DEBUG
export PYRO_LOGFILE=pyro.log
python -m Pyro4.naming -n 0.0.0.0 &
