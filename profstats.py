import pstats
import sys
proffile=sys.argv[1]
print proffile
p=pstats.Stats(proffile)
p.sort_stats('cumulative').print_stats(25)
