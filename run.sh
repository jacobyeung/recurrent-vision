parallel -j 16 "CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u run.py --dataset mnist --cuda {} --experiment {} > hi.out" ::: {0..161}
