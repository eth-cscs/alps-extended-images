#!/bin/bash

#SBATCH --nodes=1
#SBATCH --account=csstaff
#SBATCH --cpus-per-task=288
#SBATCH --time=4:00:00

TAG_NAME="$(date +%Y%m%d)"
STORE_DIR="/capstor/scratch/cscs/phimuell/.uenv-images/__ML__/"

podman build  														\
	--build-arg "BASE_IMAGE=jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.02-py3-alps6"  	\
	--build-arg "NEMORL_COMMIT=glm51-megatron-fsdp-wiring_merged"							\
	-f ./Containerfile -t "${TAG_NAME}"

if [ $? -ne 0 ]
then
	echo "Failed to build"
	exit 1
fi

CNT="I"
FILE_PAT="$(date +%Y_%m_%d)"

while true
do
	STORE_FILE="${STORE_DIR}/nemo_rl_${FILE_PAT}_${CNT}.sqsh"
	if [ ! -f "${STORE_FILE}" ]
	then
		break;
	fi
	CNT="${CNT}I"
done

echo "Store the file as '${STORE_FILE}'"
enroot import -x mount -o "${STORE_FILE}"  "podman://localhost/${TAG_NAME}"

echo "Start stripping it"
lfs migrate -c -1 -S 16M "${STORE_FILE}"





