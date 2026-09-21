#!/usr/bin/csh
#
# Batch FEFF execution (csh alternative to `mlmd-exafs run-feff`).
#
# Set DIRS to the whitespace-separated list of directories that each contain a
# feff.inp, then `source` this file. Example:
#
#     set DIRS=`find md_out/mystruct/exafs_Zn_hole1_de_0.0_s02_1.0_rc_6.0/* -type d`
#     source scripts/run_feff.csh
#
# Runs at most $max_num_processes FEFF jobs concurrently.

set FEFF_BIN=/share/feff/feff90_binaries/feff.x
set max_num_processes=32
set num_processes=0

foreach dir (${DIRS})
   cd $dir
   echo $dir
   ${FEFF_BIN} feff.inp > feff.out &   # start a new process

   set num_processes = `expr $num_processes + 1`
   if ( (`expr $num_processes % $max_num_processes`) == 0 ) then
      wait
   endif
   cd -
end

wait
