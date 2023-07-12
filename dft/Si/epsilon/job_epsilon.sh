#!/bin/sh

prefix="Si"

mpirun -np 20 pw.x < $prefix.scf.in > $prefix.scf.out
mpirun -np 20 pw.x < $prefix.nscf.in > $prefix.nscf.out
mpirun -np 20 epsilon.x < $prefix.epsilon.in > $prefix.epsilon.out