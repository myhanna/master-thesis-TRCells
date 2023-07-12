#!/bin/sh

prefix="Si"

echo "JOB STARTED. BISMILLAH ..."
mpirun -np 20 pw.x < $prefix.scf.in > $prefix.scf.out
mpirun -np 20 pw.x < $prefix.nscfband.in > $prefix.nscfband.out
mpirun -np 20 bands.x < $prefix.band.in > $prefix.band.out
rm -rf ./work/
echo "JOB DONE. ALHAMDULILLAH \(^_^)/"