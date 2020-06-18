
#!/bin/bash
EXTRACT_PATH="./DataExtraction/Stager_devel_N_24_Bing.sh"
PREPROCESS_PATH="./DataPreprocess/Stager_devel_N_24_process_netCDF.sh"
TRAINING_PATH="./Training/horovodJob.sh"
POSPROCESS_PATH = "./DataPostprocess/Stager_devel_N_24_evaluation.sh"

echo "============ Parallel Data Extraction ==========\n"

sbatch "$EXTRACT_PATH"

echo "============= Parallel Data Preprocessing =========\n "


sbatch "$PREPROCESS_PATH"


echo "============= Parallel Training ================\n"

sbatch "$TRAINING_PATH"


echo "=============Parallel Postprocessing ===============\n"

sbatch "$POSTPROCESS_PATH"



