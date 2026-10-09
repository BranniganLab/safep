#!/bin/bash

for n in {1..10}; do
  ln -sf ../../../Sample_Notebooks/Sample_Data Replica$n
done

python -m safep.AFEP_parse --replicare="Replica*" --fepoutre="*fepout" --temperature 303.15 --make_figures > test.out 2> /dev/null

if cmp -s test.out approved.out; then
  echo "Test passed"
else
  echo "Test FAILED"
fi
echo "Check FEP_general_figures.pdf. It should have 10 replicas listed (they are all identical)."
# cleanup
rm Replica*
echo "Cleaned up temporary files"